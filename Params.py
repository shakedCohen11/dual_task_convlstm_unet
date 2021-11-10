import DataHandeling
import os
from datetime import datetime
import Networks as Nets

__author__ = 'arbellea@post.bgu.ac.il'

ROOT_DATA_DIR = 'BGUSIM/Fluo-N2DH-BGUSIM'
ROOT_SAVE_DIR = 'LSTMUnet'


class ParamsBase(object):
    aws = False
    input_depth = 0
    add_tra_output = True
    send_email = False
    email_username = ' '
    email_password = " "
    receiver_email = ' '
    # parameters for experiments only
    keep_seg = 1
    remove_tra = False

    def _override_params_(self, params_dict: dict):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        this_dict = self.__class__.__dict__.keys()
        for key, val in params_dict.items():
            if key not in this_dict:
                print('Warning!: Parameter:{} not in defualt parameters'.format(key))
            setattr(self, key, val)

    pass


class CTCParams(ParamsBase):
    # --------General-------------
    dataset_name = 'Fluo-N2DH-BGUSIM'
    experiment_name = 'Train'
    dry_run = False  # Default False! Used for testing, when True no checkpoints or tensorboard outputs will be saved
    gpu_id = 0  # set -1 for CPU or GPU index for GPU.

    #  ------- Data -------
    data_provider_class = DataHandeling.CTCRAMReaderSequence2D
    root_data_dir = ROOT_DATA_DIR
    train_sequence_list = [('Train', '{:02d}'.format(s)) for s in range(1, 41)]
    val_sequence_list = [('Train', '{:02d}'.format(s)) for s in range(41, 45)]

    add_tra_output = True
    sigmoid_output = True
    dist_loss = False
    erode_dilate_tra = (0, 0)
    crop_size = (256, 256)
    batch_size = 5
    unroll_len = 4
    data_format = 'NCHW'
    train_q_capacity = val_q_capacity = 200
    num_val_threads = 1
    num_train_threads = 2

    # -------- Network Architecture ----------
    net_model = Nets.ULSTMnet2D
    net_kernel_params = {
        'down_conv_kernels': [
            [(5, 32), (5, 32)],
            [(5, 64), (5, 64)],
            [(5, 128), (5, 128)],
            [(5, 256), (5, 256)],
        ],
        'lstm_kernels': [
            [(5, 16)],
            [(5, 32)],
            [(5, 64)],
            [(5, 128)],
        ],
        'up_conv_kernels': [
            [(5, 128), (5, 128)],
            [(5, 64), (5, 64)],
            [(5, 32), (5, 32)],
            [(5, 16), (5, 16), (1, 3)],
        ],

    }

    # -------- Training ----------
    class_weights = [0.15, 0.25, 0.6]
    marker_weights = [0.5, 0.5]
    learning_rate = 1e-5
    num_iterations = 200000
    validation_interval = 2000
    print_to_console_interval = 100

    # ---------Save and Restore ----------
    load_checkpoint = False  # default is False, set to True when loading check point
    load_checkpoint_path = ''  # Used only if load_checkpoint is True
    continue_run = False  # set True for continue training the same experiment
    save_checkpoint_dir = ROOT_SAVE_DIR
    save_checkpoint_iteration = 5000
    save_checkpoint_every_N_hours = 24
    save_checkpoint_max_to_keep = 5

    # ---------Tensorboard-------------
    tb_sub_folder = 'LSTMUNet2D'
    write_to_tb_interval = 10000
    save_log_dir = ROOT_SAVE_DIR

    # ---------Debugging-------------
    profile = False
    aws = False

    def __init__(self, params_dict):
        self._override_params_(params_dict)
        if isinstance(self.gpu_id, list):

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)[1:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        self.train_data_base_folders = [(os.path.join(self.root_data_dir, ds[0]), ds[1]) for ds in
                                        self.train_sequence_list]
        self.val_data_base_folders = [(os.path.join(self.root_data_dir, ds[0]), ds[1]) for ds in
                                      self.val_sequence_list]
        self.train_data_provider = self.data_provider_class(sequence_folder_list=self.train_data_base_folders,
                                                            image_crop_size=self.crop_size,
                                                            unroll_len=self.unroll_len,
                                                            deal_with_end=0,
                                                            batch_size=self.batch_size,
                                                            queue_capacity=self.train_q_capacity,
                                                            data_format=self.data_format,
                                                            randomize=True,
                                                            return_dist=False,
                                                            num_threads=self.num_train_threads,
                                                            output_tra=self.add_tra_output,
                                                            erode_dilate_tra=self.erode_dilate_tra,
                                                            keep_sample=self.keep_seg,
                                                            remove_tra=self.remove_tra,
                                                          )
        self.val_data_provider = self.data_provider_class(sequence_folder_list=self.val_data_base_folders,
                                                          image_crop_size=self.crop_size,
                                                          unroll_len=self.unroll_len,
                                                          deal_with_end=0,
                                                          batch_size=self.batch_size,
                                                          queue_capacity=self.train_q_capacity,
                                                          data_format=self.data_format,
                                                          randomize=True,
                                                          return_dist=False,
                                                          num_threads=self.num_val_threads,
                                                          output_tra=self.add_tra_output,
                                                          erode_dilate_tra=self.erode_dilate_tra,
                                                          keep_sample=self.keep_seg,
                                                          remove_tra=self.remove_tra
                                                          )

        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S') if not self.aws else ''
        base_path = None
        if self.aws:
            self.continue_run = self.load_checkpoint = True

            base_path = self.load_checkpoint_path = os.path.join(self.save_log_dir, self.tb_sub_folder,
                                                                 self.experiment_name)
            os.makedirs(base_path, exist_ok=True)
            os.makedirs(os.path.join(base_path, 'tf_ckpts'), exist_ok=True)

        if self.load_checkpoint:
            if os.path.isdir(self.load_checkpoint_path) and not (self.load_checkpoint_path.endswith('tf_ckpts') or
                                                                 self.load_checkpoint_path.endswith('tf_ckpts/')):
                base_path = self.load_checkpoint_path
                self.load_checkpoint_path = os.path.join(self.load_checkpoint_path, 'tf_ckpts')
            elif os.path.isdir(self.load_checkpoint_path):
                base_path = os.path.dirname(self.load_checkpoint_path)
            else:

                base_path = os.path.dirname(os.path.dirname(self.load_checkpoint_path))
        if self.continue_run:
            self.experiment_log_dir = self.experiment_save_dir = base_path
        else:  # if new run set folders path for experiment
            self.experiment_log_dir = os.path.join(self.save_log_dir, self.tb_sub_folder, self.dataset_name, self.experiment_name,
                                                   now_string)
            self.experiment_save_dir = os.path.join(self.save_checkpoint_dir, self.tb_sub_folder, self.dataset_name, self.experiment_name,
                                                    now_string)

        # create relevant folders if they don't exist
        if not self.dry_run:
            os.makedirs(self.experiment_log_dir, exist_ok=True)
            os.makedirs(self.experiment_save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.experiment_save_dir,'tf_ckpts') ,exist_ok=True)
            os.makedirs(os.path.join(self.experiment_log_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.experiment_log_dir, 'val'), exist_ok=True)
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if self.profile:
            if 'CUPTI' not in os.environ['LD_LIBRARY_PATH']:
                os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda/extras/CUPTI/lib64/'


class CTCParams3DSlice(ParamsBase):
    # --------General-------------
    dataset_name = 'Fluo-N3DH-BGUSIM'
    experiment_name = 'Train'
    dry_run = False  # Default False! Used for testing, when True no checkpoints or tensorboard outputs will be saved
    gpu_id = 0  # set -1 for CPU or GPU index for GPU.

    #  ------- Data -------
    data_provider_class = DataHandeling.CTCRAMReaderSequence3DSlice
    root_data_dir = ROOT_DATA_DIR
    train_sequence_list = [('Train', '{:02d}'.format(s), True) for s in range(1, 31)]
    val_sequence_list = [('Train', '{:02d}'.format(s), True) for s in range(31, 35)]

    load_to_ram = False
    add_tra_output = True
    sigmoid_output = True
    dist_loss = False
    erode_dilate_tra = (0, 0)
    crop_size = (256, 256)
    batch_size = 5
    unroll_len = 4
    input_depth = 1
    data_format = 'NCHW'
    train_q_capacity = val_q_capacity = 200
    num_val_threads = 1
    num_train_threads = 3


    # -------- Network Architecture ----------
    net_model = Nets.ULSTMnet2D
    net_kernel_params = {
        'down_conv_kernels': [
            [(5, 32), (5, 32)],
            [(5, 64), (5, 64)],
            [(5, 128), (5, 128)],
            [(5, 256), (5, 256)],
        ],
        'lstm_kernels': [
            [(5, 16)],
            [(5, 32)],
            [(5, 64)],
            [(5, 128)],
        ],
        'up_conv_kernels': [
            [(5, 128), (5, 128)],
            [(5, 64), (5, 64)],
            [(5, 32), (5, 32)],
            [(5, 16), (5, 16), (1, 3)],
        ],

    }

    # -------- Training ----------
    class_weights = [0.15, 0.25, 0.6]
    learning_rate = 1e-5
    num_iterations = 300000
    validation_interval = 1000
    print_to_console_interval = 100

    # ---------Save and Restore ----------
    load_checkpoint = False
    load_checkpoint_path = ' '  # Used only if load_checkpoint is True
    continue_run = False
    save_checkpoint_dir = ROOT_SAVE_DIR
    save_checkpoint_iteration = 500
    save_checkpoint_every_N_hours = 24
    save_checkpoint_max_to_keep = 5

    # ---------Tensorboard-------------
    tb_sub_folder = 'LSTMUNet3DSlice'
    write_to_tb_interval = 1000
    save_log_dir = ROOT_SAVE_DIR

    # ---------Debugging-------------
    profile = False
    aws = False

    def __init__(self, params_dict):

        self._override_params_(params_dict)
        gpu_num = 1
        if isinstance(self.gpu_id, list):

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)[1:-1]
            gpu_num = len(self.gpu_id)
            self.devices = ['/gpu:{}'.format(gid) for gid in self.gpu_id]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
            self.gpu_id = int(self.gpu_id)
            self.devices = ['/gpu:{}'.format(self.gpu_id)]

        self.batch_size = self.batch_size*gpu_num
        self.center_slice = int(self.crop_size[0]/2)

        self.train_data_base_folders = [(os.path.join(self.root_data_dir, ds[0]), ds[1], ds[2]) for ds in
                                        self.train_sequence_list]
        self.val_data_base_folders = [(os.path.join(self.root_data_dir, ds[0]), ds[1], ds[2]) for ds in
                                      self.val_sequence_list]
        self.train_data_provider = self.data_provider_class(sequence_folder_list=self.train_data_base_folders,
                                                            image_crop_size=self.crop_size,
                                                            unroll_len=self.unroll_len,
                                                            deal_with_end=0,
                                                            batch_size=self.batch_size,
                                                            queue_capacity=self.train_q_capacity,
                                                            data_format=self.data_format,
                                                            randomize=True,
                                                            return_dist=False,
                                                            num_threads=self.num_train_threads,
                                                            depth=self.input_depth,
                                                            load_to_ram= self.load_to_ram,
                                                            output_tra=self.add_tra_output,
                                                            erode_dilate_tra=self.erode_dilate_tra
                                                            )
        self.val_data_provider = self.data_provider_class(sequence_folder_list=self.val_data_base_folders,
                                                          image_crop_size=self.crop_size,
                                                          unroll_len=self.unroll_len,
                                                          deal_with_end=0,
                                                          batch_size=self.batch_size,
                                                          queue_capacity=self.train_q_capacity,
                                                          data_format=self.data_format,
                                                          randomize=True,
                                                          return_dist=False,
                                                          num_threads=self.num_val_threads,
                                                          depth=self.input_depth,
                                                          load_to_ram=self.load_to_ram,
                                                            output_tra=self.add_tra_output,
                                                            erode_dilate_tra=self.erode_dilate_tra
                                                          )

        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S') if not self.aws else ''
        base_path = None
        if self.aws:
            self.load_checkpoint = True
            self.continue_run = True
            base_path = self.load_checkpoint_path = os.path.join(self.save_log_dir, self.tb_sub_folder,
                                                                 self.experiment_name)
            os.makedirs(base_path, exist_ok=True)
            os.makedirs(os.path.join(base_path, 'tf_ckpts'), exist_ok=True)

        if self.load_checkpoint:
            if os.path.isdir(self.load_checkpoint_path) and not (self.load_checkpoint_path.endswith('tf_ckpts') or
                                                                 self.load_checkpoint_path.endswith('tf_ckpts/')):
                base_path = self.load_checkpoint_path
                self.load_checkpoint_path = os.path.join(self.load_checkpoint_path, 'tf_ckpts')
            elif os.path.isdir(self.load_checkpoint_path):
                base_path = os.path.dirname(self.load_checkpoint_path)
            else:

                base_path = os.path.dirname(os.path.dirname(self.load_checkpoint_path))
        if self.continue_run:
            self.experiment_log_dir = self.experiment_save_dir = base_path
        else:
            self.experiment_log_dir = os.path.join(self.save_log_dir, self.tb_sub_folder, self.dataset_name,
                                                   self.experiment_name, now_string)
            self.experiment_save_dir = os.path.join(self.save_checkpoint_dir, self.tb_sub_folder, self.dataset_name,
                                                    self.experiment_name, now_string)

        if not self.dry_run:
            os.makedirs(self.experiment_log_dir, exist_ok=True)
            os.makedirs(self.experiment_save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.experiment_save_dir, 'tf_ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.experiment_log_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.experiment_log_dir, 'val'), exist_ok=True)
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if self.profile:
            if 'CUPTI' not in os.environ['LD_LIBRARY_PATH']:
                os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda/extras/CUPTI/lib64/'


class CTCInferenceParams(ParamsBase):

    gpu_id = 0  # for CPU ise -1 otherwise gpu id
    seq = 45
    no_tra = False
    model_path = 'LSTMUNet2D/Fluo-N2DH-BGUSIM/Train'
    ckpt_path = 'model.ckpt' # defulte ckpt path
    output_path = 'BGUSIM/Fluo-N2DH-BGUSIM/Test/{}_RES'.format(seq)
    sequence_path = 'BGUSIM/Fluo-N2DH-BGUSIM/Test/{}'.format(seq)
    filename_format = 't*.tif'  # default format for CTC
    digit_4 = False  # for 4 digit format datasets
    edge_thresh = 0.3
    data_reader_time = DataHandeling.CTCInferenceReaderTime  # for test time augmentations
    data_reader = DataHandeling.CTCInferenceReader
    # field of view prameters from website
    FOV = 0
    data_format = 'NCHW'  # 'NCHW' or 'NHWC'
    min_cell_size = 100
    max_cell_size = 100000
    must_have_cnt = False
    sigmoid_output = True

    edge_dist = 1
    pre_sequence_frames = 4  # LSTM history for first frames, done using mirror padding (in time) of this size
    centers_sigmoid_threshold = 0.5  # for binrization of TRA output (soft to BW image)
    min_center_size = 33

    # ---------Debugging---------

    dry_run = False
    save_intermediate = False  # only save intermediate images if needed
    save_intermediate_path = os.path.join(model_path, 'Outputs', os.path.basename(sequence_path))

    def __init__(self, params_dict: dict = None):

        if params_dict is not None:
            self._override_params_(params_dict)
        if isinstance(self.gpu_id, list):

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)[1:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if not self.dry_run:
            os.makedirs(self.output_path, exist_ok=True)
            if self.save_intermediate:
                now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                self.save_intermediate_path = os.path.join(self.save_intermediate_path, now_string)
                self.save_intermediate_vis_path = os.path.join(self.save_intermediate_path, 'Softmax')
                self.save_intermediate_label_path = os.path.join(self.save_intermediate_path, 'Labels')
                self.save_intermediate_label_color_path = os.path.join(self.save_intermediate_path, 'Color')
                self.save_intermediate_centers_path = os.path.join(self.save_intermediate_path, 'Centers')
                self.save_intermediate_centers_hard_path = os.path.join(self.save_intermediate_path, 'CentersHard')
                self.save_intermediate_contour_path = os.path.join(self.save_intermediate_path, 'Contours')
                os.makedirs(self.save_intermediate_path, exist_ok=True)
                os.makedirs(self.save_intermediate_vis_path, exist_ok=True)
                os.makedirs(self.save_intermediate_label_path, exist_ok=True)
                os.makedirs(self.save_intermediate_label_color_path, exist_ok=True)
                os.makedirs(self.save_intermediate_centers_path, exist_ok=True)
                os.makedirs(self.save_intermediate_contour_path, exist_ok=True)
                os.makedirs(self.save_intermediate_centers_hard_path, exist_ok=True)


class CTCInferenceParams3DSlice(ParamsBase):

    gpu_id = 0  # for CPU ise -1 otherwise gpu id
    seq = 35
    model_path = 'LSTMUNet3DSlice/Fluo-N3DH-BGUSIM/Train/2021-11-10_105701'
    output_path = 'BGUSIM/Fluo-N3DH-BGUSIM/Test/{:02d}_RES'.format(seq)
    sequence_path = 'BGUSIM/Fluo-N3DH-BGUSIM/Test/{:02d}'.format(seq)
    filename_format = 't*.tif'  # default format for CTC

    data_reader = DataHandeling.CTCInferenceReader3DSlice
    FOV = 0
    data_format = 'NCHW'  # 'NCHW' or 'NHWC'
    min_cell_size = 100
    max_cell_size = 100000
    edge_dist = 2
    pre_sequence_frames = 0
    centers_sigmoid_threshold = 0.5
    min_center_size = 10
    edge_thresh = 0.5
    no_tra = False
    one_object = False
    sigmoid_output = True

    # ---------Debugging---------

    dry_run = False
    save_intermediate = False
    save_intermediate_path = os.path.join(model_path, 'Outputs', os.path.basename(sequence_path))

    def __init__(self, params_dict: dict = None):

        if params_dict is not None:
            self._override_params_(params_dict)
        if isinstance(self.gpu_id, list):

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)[1:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if not self.dry_run:
            os.makedirs(self.output_path, exist_ok=True)
            if self.save_intermediate:
                now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                self.save_intermediate_path = os.path.join(self.save_intermediate_path, now_string)
                self.save_intermediate_vis_path = os.path.join(self.save_intermediate_path, 'Softmax')
                self.save_intermediate_label_path = os.path.join(self.save_intermediate_path, 'Labels')
                self.save_intermediate_contour_path = os.path.join(self.save_intermediate_path, 'Contours')
                self.save_intermediate_centers_path = os.path.join(self.save_intermediate_path, 'Centers')
                self.save_intermediate_centers_hard_path = os.path.join(self.save_intermediate_path, 'CentersHard')
                os.makedirs(self.save_intermediate_path, exist_ok=True)
                os.makedirs(self.save_intermediate_vis_path, exist_ok=True)
                os.makedirs(self.save_intermediate_label_path, exist_ok=True)
                os.makedirs(self.save_intermediate_contour_path, exist_ok=True)
                os.makedirs(self.save_intermediate_centers_path, exist_ok=True)
                os.makedirs(self.save_intermediate_centers_hard_path, exist_ok=True)




