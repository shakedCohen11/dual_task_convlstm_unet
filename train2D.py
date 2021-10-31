import argparse
import os
import pickle
# noinspection PyPackageRequirements
import tensorflow as tf
import Networks as Nets
import Params
import DataHandeling
import sys
import losses
from utils import log_print, format_exception
import requests
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import tensorflow.keras as k
# from aws import AWSError
# from aws.manage_instance import stop_this_instance, should_spot_terminate

__author__ = 'arbellea@post.bgu.ac.il'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# noinspection PyUnresolvedReferences
tf_version = tf.__version__

print(f'Using Tensorflow version {tf_version}')
if not tf_version.split('.')[0] == '2':
    raise ImportError(f'Required tensorflow version 2.x. current version is: {tf_version}')


def train(params):
    stop_spot_instance = False
    device = '/gpu:0' if params.gpu_id >= 0 else '/cpu:0'
    with tf.device(device):
        # Data input
        train_data_provider = params.train_data_provider
        val_data_provider = params.val_data_provider
        coord = tf.train.Coordinator()
        train_data_provider.start_queues(coord)
        val_data_provider.start_queues(coord)

        # Model

        model = params.net_model(params.net_kernel_params, params.data_format, False,
                                 add_binary_output=params.add_tra_output, sigmoid_output=params.sigmoid_output)

        # Losses and Metrics

        ce_loss_op = losses.WeightedCELoss(params.channel_axis + 1, params.class_weights)
        if params.add_tra_output:
            if params.sigmoid_output:
                bce = tf.nn.sigmoid_cross_entropy_with_logits
            else:
                wbce = losses.WeightedCELoss(params.channel_axis + 1, params.marker_weights, 2)

        seg_measure = losses.seg_measure(params.channel_axis + 1, three_d=False)
        train_loss = k.metrics.Mean(name='train_loss')
        train_ce_loss = k.metrics.Mean(name='train_ce_loss')
        train_bce_loss = k.metrics.Mean(name='train_bce_loss')
        train_seg_measure = k.metrics.Mean(name='train_seg_measure')
        train_accuracy = k.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        val_loss = k.metrics.Mean(name='val_loss')
        val_ce_loss = k.metrics.Mean(name='val_ce_loss')
        val_bce_loss = k.metrics.Mean(name='val_bce_loss')
        val_accuracy = k.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        val_seg_measure = k.metrics.Mean(name='val_seg_measure')

        # Save Checkpoints
        optimizer = tf.compat.v2.keras.optimizers.Adam(lr=params.learning_rate)
        ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=optimizer, net=model)
        if params.load_checkpoint:

            if os.path.isdir(params.load_checkpoint_path):
                latest_checkpoint = tf.train.latest_checkpoint(params.load_checkpoint_path)
            else:
                latest_checkpoint = params.load_checkpoint_path
            try:
                print(latest_checkpoint)
                if latest_checkpoint is None or latest_checkpoint == '':
                    log_print("Initializing from scratch.")
                else:
                    # ckpt.restore(latest_checkpoint)
                    model.load_weights(latest_checkpoint)
                    log_print("Restored from {}".format(latest_checkpoint))
                    # log_print("Restored from {}".format(os.path.join(params.model_path, 'model.ckpt')))

            except tf.errors.NotFoundError:
                raise ValueError("Could not load checkpoint: {}".format(latest_checkpoint))

        else:
            log_print("Initializing from scratch.")

        manager = tf.train.CheckpointManager(ckpt, os.path.join(params.experiment_save_dir, 'tf_ckpts'),
                                             max_to_keep=params.save_checkpoint_max_to_keep,
                                             keep_checkpoint_every_n_hours=params.save_checkpoint_every_N_hours)

        @tf.function
        def train_n(data_provider, model_):
            n = min(1, params.print_to_console_interval)
            softmax = predictions = loss = image = seg = tra = tra_pred = None
            for _ in range(n):
                if params.add_tra_output:
                    image, seg, _, is_last_b, tra = data_provider.get_batch()
                    softmax, predictions, loss, tra_pred = train_step(image, seg, model_, tra)
                else:

                    image, seg, _, is_last_b = data_provider.get_batch()
                    softmax, predictions, loss, tra_pred = train_step(image, seg, model_)
                    tra = tra_pred = None
                model_.reset_states_per_batch(is_last_b)
            return softmax, predictions, loss, image, seg, tra, tra_pred

        def train_step(image, label, model_, tra_label=None):
            with tf.GradientTape() as tape:
                if params.add_tra_output:
                    predictions, softmax, tra_pred, tra_pred_logits = model_(image, True)
                    if params.sigmoid_output:
                        if params.dist_loss:
                            bce_loss = losses.distance_loss(tra_label, tra_pred)
                        else:
                            bce_loss_pixel = bce(tf.minimum(tra_label, 1), tra_pred_logits)
                            valid = tf.cast(tf.greater_equal(tra_label, 0), tf.float32)
                            bce_loss_valid = bce_loss_pixel*valid
                            bce_loss = tf.reduce_sum(bce_loss_valid)/(tf.reduce_sum(valid) + 0.0000001)

                    else:
                        bce_loss = wbce(tra_label, tra_pred_logits)

                else:
                    predictions, softmax = model_(image, True)
                    tra_pred = None
                    bce_loss = 0
                ce_loss = ce_loss_op(label, predictions)
                loss = ce_loss + bce_loss
            gradients = tape.gradient(loss, model_.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_.trainable_variables))

            ckpt.step.assign_add(1)
            train_loss(loss)
            train_ce_loss(ce_loss)
            train_bce_loss(bce_loss)
            seg_value = seg_measure(label, predictions)
            if params.channel_axis == 1:
                predictions = tf.transpose(predictions, (0, 1, 3, 4, 2))
                label = tf.transpose(label, (0, 1, 3, 4, 2))
            train_accuracy(label, predictions)
            train_seg_measure(seg_value)
            return softmax, predictions, loss, tra_pred

        @tf.function
        def val_step(image, label, tra_label=None):
            tra_pred = None
            if params.add_tra_output:
                if params.sigmoid_output:
                    predictions, softmax, tra_pred, tra_pred_logits = model(image, False)
                    bce_loss = bce(tf.minimum(tra_label,1), tra_pred_logits)
                    valid = tf.cast(tf.greater_equal(tra_label, 0), tf.float32)
                    bce_loss = tf.reduce_sum(bce_loss*valid)/(tf.reduce_sum(valid) + 0.0000001)
                else:
                    predictions, softmax, tra_pred, tra_pred_logits = model(image, False)
                    bce_loss = wbce(tf.minimum(tra_label, 1), tra_pred_logits)

            else:
                predictions, softmax = model(image, False)
                bce_loss = 0

            t_loss = ce_loss_op(label, predictions) + bce_loss

            val_loss(t_loss)
            val_ce_loss(ce_loss_op(label, predictions))
            val_bce_loss(bce_loss)
            seg_value = seg_measure(label, predictions)
            if params.channel_axis == 1:
                predictions = tf.transpose(predictions, (0, 1, 3, 4, 2))
                label = tf.transpose(label, (0, 1, 3, 4, 2))
            val_accuracy(label, predictions)
            val_seg_measure(seg_value)
            if tra_pred is not None:
                return softmax, predictions, t_loss, tra_pred
            return softmax, predictions, t_loss

        train_summary_writer = val_summary_writer = train_scalars_dict = val_scalars_dict = None
        if not params.dry_run:
            train_log_dir = os.path.join(params.experiment_log_dir, 'train')
            val_log_dir = os.path.join(params.experiment_log_dir, 'val')
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

            val_summary_writer = tf.summary.create_file_writer(val_log_dir)
            train_scalars_dict = {'Total_Loss': train_loss, 'Seg_CE_Loss': train_ce_loss,
                                  'Tra_BCE_loss': train_bce_loss, 'SEG': train_seg_measure}
            val_scalars_dict = {'Total_Loss': val_loss,'Seg_CE_Loss': val_ce_loss,
                                'Tra_BCE_loss': val_bce_loss, 'SEG': val_seg_measure}

        def tboard(writer, step, scalar_loss_dict, images_dict):
            with tf.device('/cpu:0'):
                with writer.as_default():
                    for scalar_loss_name, scalar_loss in scalar_loss_dict.items():
                        tf.summary.scalar(scalar_loss_name, scalar_loss.result(), step=step)
                    for image_name, image in images_dict.items():
                        if params.channel_axis == 1:
                            image = tf.transpose(image, (0, 2, 3, 1))
                        tf.summary.image(image_name, image, max_outputs=1, step=step)

        def prepare_img_dict_for_tboard(im_sequence, output_sequence, seg_sequence_,
                                        tra_input=None, tra_pred=None):
            seg_onehot = tf.one_hot(tf.cast(tf.squeeze(seg_sequence_[:, -1], params.channel_axis), tf.int32), depth=3)
            display_image = im_sequence[:, -1]
            display_image = display_image - tf.reduce_min(display_image, axis=(1, 2, 3), keepdims=True)
            display_image = display_image / tf.reduce_max(display_image, axis=(1, 2, 3), keepdims=True)

            if params.channel_axis == 1:
                seg_onehot = tf.transpose(seg_onehot, (0, 3, 1, 2))
                display_image = display_image[:, params.input_depth:params.input_depth + 1]
            else:
                display_image = display_image[:, :, :, params.input_depth:params.input_depth + 1]
            print('Display image shape: ', display_image.shape)
            imgs_dict = {'Image': display_image,
                         'GT': seg_onehot,
                         'Output': output_sequence[:, -1]}
            if params.add_tra_output:
                imgs_dict['TraGT'] = tra_input[:, -1]
                imgs_dict['Tra'] = tra_pred[:, -1]
            return imgs_dict

        template = '{}: Step {}, Loss: {}, Accuracy: {}'
        err_mesg = ''
        err = None
        try:
            end_on_err = False
            # if True:
            val_states = model.get_states()
            if not params.dry_run:
                log_print('Saving Model of inference:')
                model_fname = os.path.join(params.experiment_save_dir, 'model.ckpt'.format(int(ckpt.step)))
                model.save_weights(model_fname, save_format='tf')
                with open(os.path.join(params.experiment_save_dir, 'model_params.pickle'), 'wb') as fobj:
                    pickle.dump({'name': model.__class__.__name__, 'params': (params.net_kernel_params,),
                                 'input_depth': params.input_depth, 'add_binary_output': params.add_tra_output},
                                fobj, protocol=pickle.HIGHEST_PROTOCOL)
                log_print('Saved Model to file: {}'.format(model_fname))
            while ckpt.step < params.num_iterations:
                # if params.aws:
                #     if should_spot_terminate():
                #         raise AWSError('Quitting Spot Instance Gracefully')
                try:
                    (train_output_sequence, train_predictions, train_loss_value, image_sequence,
                     seg_sequence, train_tra_input, train_tra_pred) = train_n(train_data_provider, model)
                except tf.errors.OutOfRangeError:
                    break

                # image_sequence, seg_sequence, _, is_last_batch = train_data_provider.get_batch()
                # if params.profile:
                #     tf.summary.trace_on(graph=True, profiler=True)
                #  = train_step(image_sequence, seg_sequence)
                # q_stats = [qs().numpy() for qs in params.train_data_provider.q_stat_list]
                # print(q_stats)
                if params.profile:
                    with train_summary_writer.as_default():
                        tf.summary.trace_export('train_step', step=int(ckpt.step),
                                                profiler_outdir=params.experiment_log_dir)
                # model.reset_states_per_batch(is_last_batch)  # reset states for sequences that ended

                if not int(ckpt.step) % params.write_to_tb_interval:
                    if not params.dry_run:
                        train_imgs_dict = prepare_img_dict_for_tboard(image_sequence, train_output_sequence,
                                                                      seg_sequence, train_tra_input, train_tra_pred)
                        tboard(train_summary_writer, int(ckpt.step), train_scalars_dict, train_imgs_dict)
                        log_print('Printed Training Step: {} to Tensorboard'.format(int(ckpt.step)))
                    else:
                        log_print("WARNING: dry_run flag is ON! Not saving checkpoints or tensorboard data")

                if int(ckpt.step) % params.save_checkpoint_iteration == 0 or int(ckpt.step) == params.num_iterations:
                    if not params.dry_run:
                        save_path = manager.save(int(ckpt.step))
                        log_print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                        model_fname = os.path.join(params.experiment_save_dir, 'model.ckpt'.format(int(ckpt.step)))
                        log_print('Saving Model of inference:')
                        model.save_weights(model_fname, save_format='tf')
                    else:
                        log_print("WARNING: dry_run flag is ON! Mot saving checkpoints or tensorboard data")
                if not int(ckpt.step) % params.print_to_console_interval:
                    log_print(template.format('Training', int(ckpt.step),
                                              train_loss.result(),
                                              train_accuracy.result() * 100))
                    log_print('CE train loss {}, BCE train loss {}'.format(train_ce_loss.result(),
                                                                           train_bce_loss.result()))

                if not int(ckpt.step) % params.validation_interval:
                    train_states = model.get_states()
                    model.set_states(val_states)

                    if params.add_tra_output:
                        (val_image_sequence, val_seg_sequence, _, val_is_last_batch, val_tra_sequence,
                         ) = val_data_provider.get_batch()
                        try:
                            val_output_sequence, val_predictions, val_loss_value, val_tra_pred = val_step(
                                val_image_sequence,
                                val_seg_sequence,
                                val_tra_sequence)
                        except tf.errors.OutOfRangeError:
                            break

                    else:
                        val_tra_sequence = val_tra_pred = None
                        (val_image_sequence, val_seg_sequence, _, val_is_last_batch,
                         ) = val_data_provider.get_batch()
                        val_output_sequence, val_predictions, val_loss_value = val_step(val_image_sequence,
                                                                                        val_seg_sequence)
                    model.reset_states_per_batch(val_is_last_batch)  # reset states for sequences that ended
                    if not params.dry_run:
                        val_imgs_dict = prepare_img_dict_for_tboard(val_image_sequence, val_output_sequence,
                                                                    val_seg_sequence, val_tra_sequence, val_tra_pred)
                        tboard(val_summary_writer, int(ckpt.step), val_scalars_dict, val_imgs_dict)
                        log_print('Printed Validation Step: {} to Tensorboard'.format(int(ckpt.step)))
                    else:
                        log_print("WARNING: dry_run flag is ON! Not saving checkpoints or tensorboard data")

                    log_print(template.format('Validation', int(ckpt.step),
                                              val_loss.result(),
                                              val_accuracy.result() * 100))
                    val_states = model.get_states()
                    model.set_states(train_states)
                    if coord.should_stop():
                        stop_spot_instance = True
                        coord.raise_requested_exception()
            if coord.should_stop():
                stop_spot_instance = True
                coord.raise_requested_exception()
            stop_spot_instance = True

        # except AWSError as err:
        #     if not params.dry_run:
        #         log_print('Saving Model Before closing due to error: {}'.format(str(err)))
        #         save_path = manager.save(int(ckpt.step))
        #         log_print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        #         # raise err
        #     end_on_err = True
        #     err_mesg = 'AWS spot shutdown'
        #     stop_spot_instance = False
        except KeyboardInterrupt as err:

            if not params.dry_run:
                log_print('Saving Model Before closing due to error: {}'.format(str(err)))
                save_path = manager.save(int(ckpt.step))
                log_print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                # raise err
            end_on_err = True
            err_mesg = ''
            params.send_email = False
            stop_spot_instance = False

        except Exception as err:
            #
            # raise err
            end_on_err = True
            err_mesg = format_exception(err)
            stop_spot_instance = True
            # raise err
        finally:

            if not params.dry_run:
                log_print('Saving Model of inference:')
                model_fname = os.path.join(params.experiment_save_dir, 'model.ckpt'.format(int(ckpt.step)))
                model.save_weights(model_fname, save_format='tf')
                # with open(os.path.join(params.experiment_save_dir, 'model_params.pickle'), 'wb') as fobj:
                #     pickle.dump({'name': model.__class__.__name__, 'params': (params.net_kernel_params,)},
                #                 fobj, protocol=pickle.HIGHEST_PROTOCOL)
                log_print('Saved Model to file: {}'.format(model_fname))
            else:
                log_print('WARNING: dry_run flag is ON! Not Saving Model')
            log_print('Closing gracefully')
            coord.request_stop()
            if params.send_email and not params.dry_run:
                send_mail(params.email_username, params.email_password, params.receiver_email,
                          params.experiment_save_dir, send_err=end_on_err, err_msg=err_mesg)

            try:
                coord.join()
            finally:
                log_print('Done')
                print(stop_spot_instance)
                # if params.aws and stop_spot_instance:
                #     stop_this_instance()
                #     if err is not None:
                #         raise err


def send_mail(sender_email, password, receiver_email, run_path, send_err=True, err_msg=''):
    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls
    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    server = None
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.ehlo()  # Can be omitted
        server.starttls(context=context)  # Secure the connection
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        if send_err:
            message = """\
                The run: {} quit because of error.""".format(run_path)
        else:
            message = """\
                    The run: {} ended sucesfuly.""".format(run_path)
        print(message)
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = 'Email from LSTM-UNet Train'
        part1 = MIMEText(message, 'plain')
        part2 = MIMEText(err_msg, 'plain')
        msg.attach(part1)
        msg.attach(part2)
        print(msg.keys())
        server.sendmail(sender_email, receiver_email, msg.as_string())
    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        if server is not None:
            server.quit()


# noinspection DuplicatedCode
if __name__ == '__main__':

    class AddNets(argparse.Action):
        import Networks as Nets

        def __init__(self, option_strings, dest, **kwargs):
            super(AddNets, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            nets = [getattr(Nets, v) for v in values]
            setattr(namespace, self.dest, nets)


    class AddReader(argparse.Action):

        def __init__(self, option_strings, dest, **kwargs):
            super(AddReader, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            reader = getattr(DataHandeling, values)
            setattr(namespace, self.dest, reader)


    class AddDatasets(argparse.Action):

        def __init__(self, option_strings, dest, *args, **kwargs):

            super(AddDatasets, self).__init__(option_strings, dest, *args, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):

            if len(values) % 2:
                raise ValueError("dataset values should be of length 2*N where N is the number of datasets")
            datastets = []
            for i in range(0, len(values), 2):
                datastets.append((values[i], (values[i + 1])))
            setattr(namespace, self.dest, datastets)

    arg_parser = argparse.ArgumentParser(description='Run Train LSTMUnet Segmentation')
    arg_parser.add_argument('-n', '--experiment_name', dest='experiment_name', type=str,
                            help="Name of experiment")
    arg_parser.add_argument('--gpu_id', dest='gpu_id', type=int,
                            help="Visible GPU: example, 0")
    arg_parser.add_argument('--dry_run', dest='dry_run', action='store_const', const=True,
                            help="Do not write any outputs: for debugging only")
    arg_parser.add_argument('--profile', dest='profile', type=bool,
                            help="Write profiling data to tensorboard. For debugging only")
    arg_parser.add_argument('--root_data_dir', dest='root_data_dir', type=str,
                            help="Root folder containing training data")
    arg_parser.add_argument('--data_provider_class', dest='data_provider_class', type=str, action=AddReader,
                            help="Type of data provider")
    arg_parser.add_argument('--dataset', dest='train_sequence_list', type=str, action=AddDatasets, nargs='+',
                            help="Datasets to run. string of pairs: DatasetName, SequenceNumber")
    arg_parser.add_argument('--val_dataset', dest='val_sequence_list', type=str, action=AddDatasets, nargs='+',
                            help="Datasets to run. string of pairs DatasetName, SequenceNumber")
    arg_parser.add_argument('--net_types', dest='net_types', type=int, nargs='+', action=AddNets,
                            help="Type of nets")
    arg_parser.add_argument('--add_tra_output', dest='add_tra_output', action='store_const', const=True,
                            help="Output both segmentation and centroid estimation")
    arg_parser.add_argument('--erode_dilate_tra', dest='erode_dilate_tra', type=int, nargs=2,
                            help="kernel size for the erosion (-) or dilation (+) of the tra data")
    arg_parser.add_argument('--crop_size', dest='crop_size', type=int, nargs=2,
                            help="crop size for y and x dimensions: example: 160 160")
    arg_parser.add_argument('--train_q_capacity', dest='train_q_capacity', type=int,
                            help="Capacity of training queue")
    arg_parser.add_argument('--val_q_capacity', dest='val_q_capacity', type=int,
                            help="Capacity of validation queue")
    arg_parser.add_argument('--num_train_threads', dest='num_train_threads', type=int,
                            help="Number of train data threads")
    arg_parser.add_argument('--num_val_threads', dest='num_val_threads', type=int,
                            help="Number of validation data threads")
    arg_parser.add_argument('--data_format', dest='data_format', type=str, choices=['NCHW', 'NWHC'],
                            help="Data format NCHW or NHWC")
    arg_parser.add_argument('--batch_size', dest='batch_size', type=int,
                            help="Batch size")
    arg_parser.add_argument('--unroll_len', dest='unroll_len', type=int,
                            help="LSTM unroll length")
    arg_parser.add_argument('--num_iterations', dest='num_iterations', type=int,
                            help="Maximum number of training iterations")
    arg_parser.add_argument('--validation_interval', dest='validation_interval', type=int,
                            help="Number of iterations between validation iteration")
    arg_parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_const', const=True,
                            help="Load from checkpoint")
    arg_parser.add_argument('--load_checkpoint_path', dest='load_checkpoint_path', type=str,
                            help="path to checkpoint, used only with --load_checkpoint")
    arg_parser.add_argument('--continue_run', dest='continue_run', action='store_const', const=True,
                            help="Continue run in existing directory")
    arg_parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                            help="Learning rate")
    arg_parser.add_argument('--class_weights', dest='class_weights', type=float, nargs=3,
                            help="class weights for background, foreground and edge classes")
    arg_parser.add_argument('--save_checkpoint_dir', dest='save_checkpoint_dir', type=str,
                            help="root directory to save checkpoints")
    arg_parser.add_argument('--save_log_dir', dest='save_log_dir', type=str,
                            help="root directory to save tensorboard outputs")
    arg_parser.add_argument('--tb_sub_folder', dest='tb_sub_folder', type=str,
                            help="sub-folder to save outputs")
    arg_parser.add_argument('--save_checkpoint_iteration', dest='save_checkpoint_iteration', type=int,
                            help="number of iterations between save checkpoint")
    arg_parser.add_argument('--save_checkpoint_max_to_keep', dest='save_checkpoint_max_to_keep', type=int,
                            help="max recent checkpoints to keep")
    arg_parser.add_argument('--save_checkpoint_every_N_hours', dest='save_checkpoint_every_N_hours', type=int,
                            help="keep checkpoint every N hours")
    arg_parser.add_argument('--write_to_tb_interval', dest='write_to_tb_interval', type=int,
                            help="Interval between writes to tensorboard")
    arg_parser.add_argument('--aws', dest='aws', action='store_const', const=True,
                            help="Option for AWS")
    arg_parser.add_argument('--keep_seg', dest='keep_seg', type=float,
                            help="FOR EXPERIMENTS ONLY")
    arg_parser.add_argument('--dont_add_tra_output', dest='add_tra_output', action='store_const', const=False,
                            help="Output both segmentation and centroid estimation")
    arg_parser.add_argument('--remove_tra', dest='remove_tra', action='store_const', const=True,
                            help="FOR EXPERIMENTS ONLY")
    arg_parser.add_argument('--sigmoid_output', dest='sigmoid_output', action='store_const', const=True,
                            help="for 2 layers marker softmax output")
    arg_parser.add_argument('--dist_loss', dest='dist_loss', action='store_const', const=True,
                            help="new distance loss for markers")
    sys_args = sys.argv

    input_args = arg_parser.parse_args()
    args_dict = {key: val for key, val in vars(input_args).items() if not (val is None)}
    print(args_dict)
    params_obj = Params.CTCParams(args_dict)
    train(params_obj)
