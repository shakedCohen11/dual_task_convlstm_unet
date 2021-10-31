import argparse
import os
import pickle

# import geodesic_distance
import scipy.ndimage
import cv2
import numpy as np
# noinspection PyPackageRequirements
import tensorflow as tf
import Networks as Nets
from Params import CTCInferenceParams
from distutils.util import strtobool
import DataHandeling
import sys
from utils import log_print, get_model, bbox_crop, bbox_fill, draw_labeled_on_img
import skfmm
import seaborn as sns

__author__ = 'arbellea@post.bgu.ac.il'
if not tf.__version__.split('.')[0] == '2':
    raise ImportError(f'Required tensorflow version 2.x. current version is: {tf.__version__}')

k = tf.keras


def inference():
    # Load Model
    with open(os.path.join(params.model_path, 'model_params.pickle'), 'rb') as fobj:
        model_dict = pickle.load(fobj)
    model_cls = get_model(model_dict['name'])

    device = '/gpu:0' if params.gpu_id >= 0 else '/cpu:0'
    with tf.device(device):
        add_binary_output = model_dict.get('add_binary_output', False)
        model_dict['params'][0]['up_conv_kernels'][-1][-1] = (1, 3)
        model = model_cls(*model_dict['params'], data_format=params.data_format, pad_image=True,
                          add_binary_output=add_binary_output, sigmoid_output=params.sigmoid_output)
        # allow loading of different check points
        if params.ckpt_path == 'model.ckpt':
            model.load_weights(os.path.join(params.model_path, 'model.ckpt'))
        else:
            optimizer = tf.compat.v2.keras.optimizers.Adam(lr=1e-5)
            ckpt_path = os.path.join(params.model_path, params.ckpt_path)
            tf.train.Checkpoint(net=model, optimizer=optimizer).restore(ckpt_path).expect_partial()

        log_print("Restored from {}".format(os.path.join(params.model_path, params.ckpt_path)))
    base_out_temp_vis_fname = base_out_temp_label_fname = base_out_fname = base_out_temp_label_color_fname = None
    base_out_temp_centers_fname = base_out_temp_contour_fname = base_out_temp_centers_hard_fname = None
    if not params.dry_run:
        if params.save_intermediate_path and params.save_intermediate:
            if params.digit_4:
                base_out_temp_vis_fname = os.path.join(params.save_intermediate_vis_path, 'softmax{time:04d}.tif')
                base_out_temp_centers_fname = os.path.join(params.save_intermediate_centers_path, 'sigmoid{time:04d}.tif')
                base_out_temp_contour_fname = os.path.join(params.save_intermediate_contour_path, 'contours{time:04d}.tif')
                base_out_temp_centers_hard_fname = os.path.join(params.save_intermediate_centers_hard_path,
                                                                'centers{time:04d}.tif')
                base_out_temp_label_fname = os.path.join(params.save_intermediate_label_path, 'mask{time:04d}.tif')
                base_out_temp_label_color_fname = os.path.join(params.save_intermediate_label_color_path,
                                                               'mask_color{time:04d}.tif')
            else:
                base_out_temp_vis_fname = os.path.join(params.save_intermediate_vis_path, 'softmax{time:03d}.tif')
                base_out_temp_centers_fname = os.path.join(params.save_intermediate_centers_path, 'sigmoid{time:03d}.tif')
                base_out_temp_contour_fname = os.path.join(params.save_intermediate_contour_path, 'contours{time:03d}.tif')
                base_out_temp_centers_hard_fname = os.path.join(params.save_intermediate_centers_hard_path,
                                                                'centers{time:03d}.tif')
                base_out_temp_label_fname = os.path.join(params.save_intermediate_label_path, 'mask{time:03d}.tif')
                base_out_temp_label_color_fname = os.path.join(params.save_intermediate_label_color_path,
                                                               'mask_color{time:03d}.tif')
        if params.digit_4:
            base_out_fname = os.path.join(params.output_path, 'mask{time:04d}.tif')
        else:
            base_out_fname = os.path.join(params.output_path, 'mask{time:03d}.tif')
    dataset = params.data_reader(params.sequence_path, params.filename_format,
                                 pre_sequence_frames=params.pre_sequence_frames).dataset

    current_palette = (np.array([(0,0,0)] + sns.color_palette())*255).astype(np.uint8)
    pallet_len = len(current_palette)
    try:
        for T, image in enumerate(dataset):
            t = T - params.pre_sequence_frames  # for mirror padding in time
            image_orig = image
            image_shape = image.shape
            if len(image_shape) == 2:
                if params.data_format == 'NCHW':
                    image = tf.reshape(image, [1, 1, 1, image_shape[0], image_shape[1]]) #1 time, 1 batch size 1 cannel
                else:
                    image = tf.reshape(image, [1, 1, image_shape[0], image_shape[1], 1])
            elif len(image_shape) == 3:
                image = tf.reshape(image, [1, 1, image_shape[0], image_shape[1], image_shape[2]])
            else:
                raise ValueError()
            if add_binary_output:  # did we train with TRA
                _, image_softmax, center_sigmoid, _ = model(image, training=False)
                center_sigmoid_np = center_sigmoid.numpy().squeeze()
                # center_sigmoid_np = np.squeeze(center_sigmoid.numpy(), (0, 1))
            else:
                _, image_softmax = model(image, training=False)
                center_sigmoid_np = None

            image_softmax_np = np.squeeze(image_softmax.numpy(), (0, 1))
            if t < 0:
                continue  # for "warm up" of the network

            if not params.dry_run:
                if params.data_format == 'NHWC':
                    image_softmax_np = image_softmax_np.transpose((2,0,1))
                seg_edge = np.greater_equal(image_softmax_np[2], params.edge_thresh) # get BW image with threshold for cell edges
                seg_cell_soft = image_softmax_np[1]

                seg_cell = np.logical_and(np.equal(np.argmax(image_softmax_np, 0), 1).astype(np.float32),
                                          np.logical_not(seg_edge)) # if cell label (1) and not clear edge
                seg_edge = seg_edge.astype(np.float32)
                seg_cell = scipy.ndimage.morphology.binary_fill_holes(seg_cell).astype(np.float32)
                seg_edge = np.maximum((seg_edge - seg_cell), 0) # up date cell egdes to clear edges
                cc_out = cv2.connectedComponentsWithStats(seg_cell.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
                num_cells = cc_out[0]

                labels = cc_out[1]
                stats = cc_out[2]

                # claculte the euclidean distance of the each pixel from the cell pixells
                dist, ind = scipy.ndimage.morphology.distance_transform_edt(1 - seg_cell, return_indices=True)
                # if the edge pixel is close enough to a cell pixel make it a cell pixel else background
                # the edge gets the label of the closest cell.
                labels = labels[ind[0, :], ind[1, :]] * seg_edge * (dist < params.edge_dist) + labels

                if add_binary_output:
                    if params.sigmoid_output:
                        centers_hard = np.greater(center_sigmoid_np, params.centers_sigmoid_threshold)
                    else:
                        if params.data_format == 'NHWC':
                            center_sigmoid_np = center_sigmoid_np.transpose((2, 0, 1))
                        centers_hard = np.argmax(center_sigmoid_np, 0)
                    if params.no_tra:
                        centers_hard = np.zeros_like(centers_hard)
                    centers_cc_out = cv2.connectedComponentsWithStats(centers_hard.astype(np.uint8), connectivity=8,
                                                                      ltype=cv2.CV_32S)
                    centers_labels = centers_cc_out[1]

                    if params.must_have_cnt:
                        must_have_cnt_hard = np.greater(center_sigmoid_np, 0.01)
                        must_have_cnt_cc_out = cv2.connectedComponentsWithStats(must_have_cnt_hard.astype(np.uint8), connectivity=8,
                                                                          ltype=cv2.CV_32S)
                        must_have_cnt_labels = must_have_cnt_cc_out[1]
                else:
                    centers_labels = None
                global_num_cells = num_cells
                for n in range(1, num_cells):
                    bw = labels == n
                    if not np.any(bw):
                        continue

                    bw_crop, loc = bbox_crop(bw) # crop bounding box around the cell
                    fill_crop = scipy.ndimage.morphology.binary_fill_holes(bw_crop).astype(np.float32)
                    fill_diff = fill_crop - bw_crop
                    bw_fill = bbox_fill(bw, fill_diff, loc)  # update the holes in the original BW image
                    labels = labels + bw_fill * n  # update the original labeled image

                    if add_binary_output and not params.no_tra:
                        centers_crop = centers_labels[loc[0]:loc[1], loc[2]:loc[3]]
                        unique_centers = np.unique(centers_crop[(centers_crop*fill_crop)>0])
                        unique_centers = unique_centers[unique_centers>0]
                        if params.must_have_cnt:
                            must_have_cnt_crop = must_have_cnt_labels[loc[0]:loc[1], loc[2]:loc[3]]
                            unique_must_have_cnt = np.unique(must_have_cnt_crop[(must_have_cnt_crop * fill_crop) > 0])
                            unique_must_have_cnt = unique_must_have_cnt[unique_must_have_cnt > 0]
                        # use FM to separate cells
                        if len(unique_centers) > 1:
                            seg_cell_soft_crop = seg_cell_soft[loc[0]:loc[1], loc[2]:loc[3]]
                            seg_cell_soft_crop = seg_cell_soft_crop * fill_crop
                            order = 2
                            while True:
                                try:

                                    d_c = []
                                    d_0 = []
                                    d_1 = []
                                    for c in unique_centers:
                                        if np.sum(centers_crop == c) < params.min_center_size:
                                            continue
                                        # the T is there only to fix an error in the fmm that sometime happens.
                                        # I don't know why it happens
                                        this_c = (centers_crop == c)*fill_crop
                                        phi = np.ma.MaskedArray(1 - 2 * this_c, np.logical_not(fill_crop))
                                        if order > 0:
                                            # fast marching distance
                                            d1 = skfmm.travel_time(phi, seg_cell_soft_crop * fill_crop, order=order) * (
                                                        1 - this_c)
                                            # euclidean distance
                                            d0 = skfmm.travel_time(phi, np.ones_like(phi), order=order) * (1 - this_c)
                                        else:
                                            d1 = skfmm.distance(phi) * (
                                                         1 - this_c)
                                            d0 = 0
                                        # d1 = pyfmm.march((centers_crop == c), speed=seg_cell_soft_crop,
                                        #                  batch_size=4)[0]*fill_crop
                                        # d0 = pyfmm.march((centers_crop == c), batch_size=4)[0]*fill_crop
                                        d_c.append(d1 - d0)
                                        d_0.append(d0)
                                        d_1.append(d1)
                                    break
                                except (RuntimeError, ValueError) as err:

                                    order -= 1
                                    # raise err
                                    if order < 0:
                                        break
                                    # print('Trying fmm with T: {}'.format(T))

                            # use the delta of euclidean and fast marching to relabel the cells
                            if len(d_c) > 1:
                                d_c = np.stack(d_c, 0)
                                new_label = np.argmin(d_c, axis=0) + 1
                                new_label = new_label*fill_crop
                                skip = False
                                for nc in range(len(d_c)):
                                    new_label_cc = cv2.connectedComponentsWithStats((new_label == nc).astype(np.uint8), 8,
                                                                                    cv2.CV_32S)
                                    if new_label_cc[0] > 2:
                                        skip = True
                                        break

                                if skip:
                                    new_label = np.argmin(d_1, axis=0) + 1
                                    new_label = new_label * fill_crop

                                new_label = bbox_fill(np.zeros_like(bw, dtype=np.uint16), new_label, loc)
                                for nl in np.unique(new_label):
                                    if nl <= 1:
                                        continue
                                    global_num_cells += 1
                                    new_global_l = global_num_cells
                                    labels[new_label == nl] = new_global_l

                        if params.must_have_cnt:
                            if len(unique_must_have_cnt) < 1:
                                # discard cell if has no center
                                labels[labels == n] = 0

                # filter by fov
                if params.FOV:
                    fov_im = np.ones_like(labels)
                    fov_im[:params.FOV, :] = 0
                    fov_im[-params.FOV:, :] = 0
                    fov_im[:, :params.FOV] = 0
                    fov_im[:, -params.FOV:] = 0
                    fov_labels = labels * fov_im
                    unique_fov_labels = np.unique(fov_labels.flatten())
                    remove_ind = np.setdiff1d(np.arange(num_cells), unique_fov_labels) # remove labels that are not in the FOV
                else:
                    remove_ind = []
                if params.save_intermediate:

                    image_softmax_np = np.transpose(image_softmax_np, (1, 2, 0))
                    out_fname = base_out_temp_vis_fname.format(time=t)
                    sigoutnp_vis = np.flip(np.round(image_softmax_np * (2 ** 16 - 1)).astype(np.uint16), 2)
                    cv2.imwrite(filename=out_fname, img=sigoutnp_vis.astype(np.uint16))
                    log_print("Saved File: {}".format(out_fname))
                    if add_binary_output:
                        out_fname = base_out_temp_centers_fname.format(time=t)
                        if params.sigmoid_output:
                            cv2.imwrite(filename=out_fname, img=(center_sigmoid_np*255).astype(np.uint8))
                        else:
                            cv2.imwrite(filename=out_fname, img=(center_sigmoid_np[1]*255).astype(np.uint8))
                        log_print("Saved File: {}".format(out_fname))

                # relabel image according to FOV and blob analysis (min, max):
                labels_out = np.zeros_like(labels, dtype=np.uint16)
                # isbi_out_dict = {}
                p = 0
                for n in range(1, global_num_cells+1):
                    area = np.sum(labels == n)
                    if params.min_cell_size <= area <= params.max_cell_size and not (n in remove_ind):
                        p += 1
                        # isbi_out_dict[p] = [p, 0, 0, 0]
                        labels_out[labels == n] = p

                    else:
                        labels[labels == n] = 0
                out_fname = base_out_fname.format(time=t)
                cv2.imwrite(filename=out_fname, img=labels_out.astype(np.uint16))
                log_print("Saved File: {}".format(out_fname))
                if params.save_intermediate:
                    merge = draw_labeled_on_img(image_orig.numpy(), labels_out)
                    out_fname = base_out_temp_label_fname.format(time=t)
                    cv2.imwrite(filename=out_fname, img=labels_out.astype(np.uint16))
                    log_print("Saved File: {}".format(out_fname))
                    out_fname = base_out_temp_contour_fname.format(time=t)
                    cv2.imwrite(filename=out_fname, img=merge)
                    log_print("Saved File: {}".format(out_fname))
                    if add_binary_output:
                        out_fname = base_out_temp_centers_hard_fname.format(time=t)
                        cv2.imwrite(filename=out_fname, img=(centers_hard*255).astype(np.uint8))
                        log_print("Saved File: {}".format(out_fname))
                    out_fname = base_out_temp_label_color_fname.format(time=t)
                    mod_labels = np.mod(labels_out-1, pallet_len-1)+1
                    mod_labels[labels_out == 0] = 0
                    color_img = current_palette[mod_labels]
                    cv2.imwrite(filename=out_fname, img=color_img)
                    log_print("Saved File: {}".format(out_fname))
    except (KeyboardInterrupt, ValueError) as err:
        print('Error: {}'.format(str(err)))
        raise err

    finally:
        print('Done!')


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
                datastets.append((values[i], strtobool(values[i + 1])))
            setattr(namespace, self.dest, datastets)


    arg_parser = argparse.ArgumentParser(description='Run Inference LSTMUnet Segmentation')
    arg_parser.add_argument('--gpu_id', dest='gpu_id', type=int,
                            help="Visible GPUs: example, '0', use -1 for CPU")
    arg_parser.add_argument('--model_path', dest='model_path', type=str,
                            help="Path to trained model generated by train2D.py, folder should contain model.ckpt.*")
    arg_parser.add_argument('--ckpt_path', dest='ckpt_path', type=str,
                            help="relative path to ckpt")
    arg_parser.add_argument('--sequence_path', dest='sequence_path', type=str,
                            help="Path to sequence images. Folder should contain image files")
    arg_parser.add_argument('--output_path', dest='output_path', type=str,
                            help="Path to save output images.")
    arg_parser.add_argument('--filename_format', dest='filename_format', type=str,
                            help="Format of file using wildcard (*) to indicate timestep. Default: 't*.tif'")
    arg_parser.add_argument('--data_format', dest='data_format', type=str, choices=['NCHW', 'NWHC'],
                            help="Data format NCHW or NHWC")
    arg_parser.add_argument('--min_cell_size', dest='min_cell_size', type=int,
                            help="Minimum cell size")
    arg_parser.add_argument('--fov', dest='FOV', type=int,
                            help="Feild of veiw for detection")
    arg_parser.add_argument('--max_cell_size', dest='max_cell_size', type=int,
                            help="Maximum cell size")
    arg_parser.add_argument('--num_iterations', dest='num_iterations', type=int,
                            help="Maximum number of training iterations")
    arg_parser.add_argument('--edge_dist', dest='edge_dist', type=int,
                            help="Maximum edge width to add to cell object")
    arg_parser.add_argument('--edge_thresh', dest='edge_thresh', type=float,
                            help="Threshold for edge detection")
    arg_parser.add_argument('--centers_sigmoid_threshold', dest='centers_sigmoid_threshold', type=float,
                            help="Threshold for center marker detection")
    arg_parser.add_argument('--min_center_size', dest='min_center_size', type=int,
                            help="Minimum size for center marker")
    arg_parser.add_argument('--pre_sequence_frames', dest='pre_sequence_frames', type=int,
                            help="Number of frames to run before sequence, uses mirror of first N frames.")
    arg_parser.add_argument('--save_intermediate', dest='save_intermediate', action='store_const', const=True,
                            help="Save intermediate files")
    arg_parser.add_argument('--dont_save_intermediate', dest='save_intermediate', action='store_const', const=False,
                            help="Do not save intermediate files")
    arg_parser.add_argument('--save_intermediate_path', dest='save_intermediate_path', type=str,
                            help="Path to save intermediate files, used only with --save_intermediate")
    arg_parser.add_argument('--dry_run', dest='dry_run', action='store_const', const=True,
                            help="Do not write any outputs: for debugging only")
    arg_parser.add_argument('--must_have_cnt', dest='must_have_cnt', action='store_const', const=True,
                            help="use cell centers to determine cells")
    arg_parser.add_argument('--digit_4', dest='digit_4', action='store_const', const=True,
                            help="use 4 digit format in saving the results")
    arg_parser.add_argument('--sigmoid_output', dest='sigmoid_output', action='store_const', const=True,
                            help="for 2 layers marker softmax output")
    arg_parser.add_argument('--no_tra', dest='no_tra', action='store_const', const=True,
                            help="no tra use")

    sys_args = sys.argv

    input_args = arg_parser.parse_args()
    args_dict = {key: val for key, val in vars(input_args).items() if not (val is None)}
    params = CTCInferenceParams(args_dict)
    tf_eps = tf.constant(1E-8, name='epsilon')
    try:
        inference()
    finally:
        log_print('Done')
