import random
import tensorflow as tf
import os
import glob
import cv2
import queue
import threading
import numpy as np
import pickle
import utils
import time
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import grey_dilation, grey_erosion, affine_transform  # measurements, grey_erosion
from scipy.ndimage.measurements import center_of_mass

__author__ = 'assafarbelle'


class CTCRAMReaderSequence2D(object):
    def __init__(self, sequence_folder_list, image_crop_size=(128, 128), unroll_len=7, deal_with_end=0, batch_size=4,
                 queue_capacity=32, num_threads=3,
                 data_format='NCHW', randomize=True, return_dist=False, keep_sample=1, remove_tra=False, elastic_augmentation=True,
                 output_tra=False, erode_dilate_tra=(0,0)):
        if not isinstance(image_crop_size, tuple):
            image_crop_size = tuple(image_crop_size)
        self.coord = None
        self.unroll_len = unroll_len
        self.sequence_data = {}
        self.sequence_folder_list = sequence_folder_list
        self.elastic_augmentation = elastic_augmentation
        self.sub_seq_size = image_crop_size
        self.dist_sub_seq_size = (2,) + image_crop_size
        self.deal_with_end = deal_with_end
        self.batch_size = batch_size
        self.queue_capacity = queue_capacity
        self.data_format = data_format
        self.randomize = randomize
        self.return_dist = None  # Deprecited
        self.num_threads = num_threads
        self.keep_sample = keep_sample
        self.remove_tra = remove_tra
        self.output_tra = output_tra
        self.erode_dilate_tra = erode_dilate_tra

        self.q_list, self.q_stat_list = self._create_queues()
        np.random.seed(1)

    @classmethod
    def unit_test(cls):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/'

        sequence_folder_list = [(os.path.join(root_dir, 'DIC-C2DH-HeLa'), '01'),
                                (os.path.join(root_dir, 'DIC-C2DH-HeLa'), '02')]
        image_crop_size = (256, 256)
        unroll_len = 7
        deal_with_end = 0
        batch_size = 4
        queue_capacity = 250
        num_threads = 2
        data_format = 'NCHW'
        randomize = True
        return_dist = False
        keep_sample = 1
        elastic_augmentation = True
        output_tra = True
        erode_dilate_tra = (25, -11)
        data = cls(sequence_folder_list, image_crop_size, unroll_len, deal_with_end, batch_size, queue_capacity,
                   num_threads, data_format, randomize, return_dist, keep_sample, elastic_augmentation, output_tra,
                   erode_dilate_tra)

        debug = True
        data.start_queues(debug=debug)
        for i in range(100):
            image_batch, seg_batch, full_seg, is_last, tra_outputs = data.get_batch()
            utils.log_print(image_batch.shape, seg_batch.shape, is_last.shape)

    def _read_sequence_to_ram_(self):
        for sequence_folder in self.sequence_folder_list:
            train_set = True
            seq = None
            sequence_folder_orig = sequence_folder
            if isinstance(sequence_folder, tuple):
                if len(sequence_folder) == 2:
                    sequence_folder, seq = sequence_folder
                elif len(sequence_folder) == 3:
                    sequence_folder, seq, train_set = sequence_folder

            utils.log_print('Reading Sequence {}: {}'.format(sequence_folder, seq))
            with open(os.path.join(sequence_folder, 'metadata_{}.pickle'.format(seq)), 'rb') as fobj:
                metadata = pickle.load(fobj)

            filename_list = metadata['filelist']
            img_size = metadata['shape']
            if len(img_size) == 3:
                img_size = img_size[1:]
            all_images = np.zeros((len(filename_list), img_size[0], img_size[1]))
            all_seg = np.zeros((len(filename_list), img_size[0], img_size[1]))
            all_full_seg = np.zeros((len(filename_list)))
            all_tra = np.zeros((len(filename_list), img_size[0], img_size[1])) if self.output_tra else None
            keep_rate = self.keep_sample
            original_size = 0
            downampled_size = 0
            remove_seg_list = np.random.rand(len(filename_list)) > keep_rate #for expirament
            # keep at least 1 seg image
            if np.sum(remove_seg_list.astype(np.int32)) == len(filename_list):
                remove_seg_list[np.random.choice(range(len(filename_list)), 1)] = False
            for t, (filename, remove_seg) in enumerate(zip(filename_list, remove_seg_list)):
                remove_tra = remove_seg if self.remove_tra else False # for partial makers expiraments
                # read and normalize image
                img = cv2.imread(os.path.join(sequence_folder, filename[0]), -1)
                if img is None:
                    raise ValueError('Could not load image: {}'.format(os.path.join(sequence_folder, filename[0])))
                img = img.astype(np.float32)
                img = (img - img.mean()) / (img.std())
                full_seg = 1 if filename[3] is True else 0
                if full_seg == 1:
                    original_size += 1

                keep_seg = 1.

                full_seg = full_seg if keep_seg else 0
                if full_seg == 1:
                    downampled_size += 1

                if filename[1] is None or not keep_seg:
                    seg = np.ones(img.shape[:2]) * (-1)
                elif not full_seg:
                    seg = cv2.imread(os.path.join(sequence_folder, filename[1]), -1)
                    if seg is None:
                        seg = np.ones(img.shape[:2]) * (-1)
                        full_seg = -1

                    else:
                        seg = seg.astype(np.float32)
                    seg[seg == 0] = -1
                else:
                    seg = cv2.imread(os.path.join(sequence_folder, filename[1]), -1)
                if self.output_tra:
                    tra = cv2.imread(os.path.join(sequence_folder, filename[2]), -1)
                    tra = np.minimum(tra, 1)

                    all_tra[t] = tra.astype(np.float32) if not remove_tra else np.ones_like(tra)*(-1.0)
                all_images[t] = img
                all_seg[t] = seg if not remove_seg else np.ones_like(seg)*(-1.0)

                all_full_seg[t] = full_seg
            if keep_rate < 1:
                print('Downsampling Training Segmentaiont with rate:{}. Original set size: {}. '
                      'Downsampled set size: {}, remove_tra {}'.format(keep_rate, original_size, downampled_size,
                                                                    self.remove_tra))

            self.sequence_data[sequence_folder_orig] = {'images': all_images, 'segs': all_seg, 'tra': all_tra,
                                                        'full_seg': all_full_seg, 'metadata': metadata}

    def _read_sequence_data(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        # if isinstance(sequence_folder, tuple):
        #     sequence_folder = sequence_folder[0]
        return self.sequence_data[sequence_folder], sequence_folder

    def _read_sequence_metadata(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        if isinstance(sequence_folder, tuple):
            sequence_folder = sequence_folder[0]
        with open(os.path.join(sequence_folder, 'metadata.pickle'), 'rb') as fobj:
            metadata = pickle.load(fobj)
        return metadata

    @staticmethod
    def _get_elastic_affine_matrix_(shape_size, alpha_affine):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
         .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
              Convolutional Neural Networks applied to Visual Document Analysis", in
              Proc. of the International Conference on Document Analysis and
              Recognition, 2003.

          Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         """
        random_state = np.random.RandomState(None)

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        affine_matrix = cv2.getAffineTransform(pts1, pts2)

        return affine_matrix, random_state

    @staticmethod
    def _get_transformed_image_(image, affine_matrix, indices, seg=False):

        shape = image.shape
        if seg:
            # padding with nearest neighbor interpolation
            trans_img = cv2.warpAffine(image, affine_matrix, shape[::-1], borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=-1, flags=cv2.INTER_NEAREST)
            trans_coord = map_coordinates(trans_img, indices, order=0, mode='constant', cval=-1).reshape(shape)
        else:
            # apply affine transform with mirror padding for image:
            trans_img = cv2.warpAffine(image, affine_matrix, shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
            trans_coord = map_coordinates(trans_img, indices, order=1, mode='reflect').reshape(shape)

        return trans_coord

    @staticmethod
    def _get_indices4elastic_transform(shape, alpha, sigma, random_state):
        dxr = random_state.rand(*shape)
        dx = gaussian_filter((dxr * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        return indices

    @staticmethod
    def _fix_transformed_segmentation(trans_seg, tra_image=False):
        # separate close blobs with the edge of transformation
        # edge is pixels that get fractional value after transofrmation
        strel = np.zeros((3, 3))
        trans_seg = np.round(trans_seg)
        # region_props = skimage.measure.regionprops(trans_seg)
        # errosion = grey_erosion(trans_seg, np.zeros(3, 3, 3))
        dilation = grey_dilation(trans_seg.astype(np.int32), structure=strel.astype(np.int8))
        bw = np.minimum(trans_seg, 1)
        if tra_image:
            bw[np.logical_and(np.not_equal(trans_seg, dilation), np.greater(dilation, 0))] = 0
        else:
            bw[np.logical_and(np.not_equal(trans_seg, dilation), np.greater(dilation, 0))] = 2

        return bw

    @staticmethod
    def _gt2dist_(gt_image):
        gt_fg = gt_image == 1
        _, labeled_gt = cv2.connectedComponents(gt_fg.astype(np.uint8))
        im_shape = gt_image.shape
        dist_1 = np.ones_like(gt_image) * (im_shape[0] + im_shape[1]) + 2.
        dist_2 = dist_1 + 1.

        for label in np.unique(labeled_gt):
            if label == 0:
                continue
            bw = np.equal(labeled_gt, label).astype(np.float32)
            bw_erode = cv2.erode(bw, np.ones((3, 3)))
            edge = np.logical_and(np.logical_not(bw_erode), bw)

            dist = distance_transform_edt(np.logical_not(edge))
            is_first_dist = np.less(dist, dist_1)
            dist_2[is_first_dist] = dist_1[is_first_dist]
            is_second_dist = np.logical_and(np.less(dist, dist_2), np.logical_not(is_first_dist))

            dist_1[is_first_dist] = dist[is_first_dist]
            dist_2[is_second_dist] = dist[is_second_dist]
        out = np.stack((dist_1, dist_2), 0)

        return out, (dist_1, dist_2)

    @staticmethod
    def _adjust_brightness_(image, delta):
        """
        Args:
        image (numpy.ndarray)
        delta
        """

        out_img = image + delta
        return out_img

    @staticmethod
    def _adjust_contrast_(image, factor):
        """
        Args:
        image (numpy.ndarray)
        factor
        """

        img_mean = image.mean()
        out_img = (image - img_mean) * factor + img_mean
        return out_img

    def _load_and_enqueue(self, q, q_stat):
        tra_crop = None
        unroll_len = self.unroll_len
        max_t_global = 1
        processed_image = np.zeros((max_t_global,) + self.sub_seq_size)
        processed_seg = np.zeros((max_t_global,) + self.sub_seq_size[:2])
        processed_tra = np.zeros((max_t_global,) + self.sub_seq_size[:2])
        full_seg_seq = np.zeros((max_t_global,))
        is_last = np.zeros((max_t_global,))
        try:
            while not self.coord.should_stop():
                seq_data, sequence_folder = self._read_sequence_data()
                img_size = seq_data['metadata']['shape']

                max_t = len(seq_data['metadata']['filelist'])

                if max_t > max_t_global:
                    max_t_global = max_t
                    processed_image = np.zeros((max_t_global,) + self.sub_seq_size)
                    processed_seg = np.zeros((max_t_global,) + self.sub_seq_size[:2])
                    processed_tra = np.zeros((max_t_global,) + self.sub_seq_size[:2])
                    full_seg_seq = np.zeros((max_t_global,))
                    is_last = np.zeros((max_t_global,))

                random_sub_sample = np.random.randint(1, 4) if self.randomize else 0
                random_reverse = np.random.randint(0, 2) if self.randomize else 0
                # use 2D gaussian distribution of indexs s.t the edges get less probability
                if img_size[0] - self.sub_seq_size[0] > 0:
                    sub_space_y = img_size[0] - self.sub_seq_size[0]
                    mu_y = sub_space_y / 2
                    sigma_y = sub_space_y / 6.4
                    crop_y = np.random.normal(mu_y, sigma_y) if self.randomize else 0
                else:
                    crop_y = 0
                crop_y = int(crop_y)
                crop_y_stop = crop_y + self.sub_seq_size[0]
                # get index back to bound
                if crop_y < 0:
                    # shift up
                    crop_y_stop = crop_y_stop + abs(crop_y)
                    crop_y = 0
                if crop_y_stop > img_size[0]:
                    # shift down
                    crop_y = crop_y - (crop_y_stop - img_size[0])
                    crop_y_stop = img_size[0]

                if img_size[1] - self.sub_seq_size[1] > 0:
                    sub_space_x = img_size[1] - self.sub_seq_size[1]
                    mu_x = sub_space_x / 2
                    sigma_x = sub_space_x / 6.4
                    crop_x = np.random.normal(mu_x, sigma_x) if self.randomize else 0
                else:
                    crop_x = 0
                crop_x = int(crop_x)
                crop_x_stop = crop_x + self.sub_seq_size[1]
                # get index back to bound
                if crop_x < 0:
                    # shift right
                    crop_x_stop = crop_x_stop + abs(crop_x)
                    crop_x = 0
                if crop_x_stop > img_size[1]:
                    # shift left
                    crop_x = crop_x - (crop_x_stop - img_size[1])
                    crop_x_stop = img_size[1]

                flip = np.random.randint(0, 2, 2) if self.randomize else [0, 0]
                rotate = np.random.randint(0, 4) if self.randomize else 0
                if self.elastic_augmentation:
                    # set boundry size acording to img size
                    boundary_x = int(img_size[1]*0.08)
                    boundary_y = int(img_size[0]*0.08)
                    affine_size = (self.sub_seq_size[0]+2*boundary_y, self.sub_seq_size[1]+2*boundary_x)

                    affine_matrix, random_state = self._get_elastic_affine_matrix_(affine_size,
                                                                                   affine_size[1] * 0.08)
                    indices = self._get_indices4elastic_transform(affine_size, affine_size[1] * 2,
                                                                  affine_size[1] * 0.15,
                                                                  random_state)
                else:
                    affine_matrix = indices = None

                filename_idx = list(range(len(seq_data['metadata']['filelist'])))

                if random_reverse:
                    filename_idx.reverse()
                if random_sub_sample:
                    filename_idx = filename_idx[::random_sub_sample]
                seq_len = len(filename_idx)
                remainder = seq_len % unroll_len

                if remainder:
                    if self.deal_with_end == 0:
                        filename_idx = filename_idx[:-remainder]
                    elif self.deal_with_end == 1:
                        filename_idx += filename_idx[-2:-unroll_len + remainder - 2:-1]
                    elif self.deal_with_end == 2:
                        filename_idx += filename_idx[-1:] * (unroll_len - remainder)

                boundary_flag = False
                # if we randomed a crop on the boundary of the image we dont use affine transform
                if (crop_x < boundary_x or crop_x_stop > img_size[1] - boundary_x or
                        crop_y < boundary_y or crop_y_stop > img_size[0] - boundary_y
                        or not self.elastic_augmentation):
                    boundary_flag = True
                    img_crops = seq_data['images'][:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                    img_max = seq_data['images'].max()
                    seg_crops = seq_data['segs'][:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                    if self.output_tra:
                        tra_crops = seq_data['tra'][:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                    else:
                        tra_crops = None
                else:
                    # apply affine transform on lager segment and then crop, this is done to avoid unwanted artifacts
                    # from the mirror image padding
                    # move upper left corner by boundry size:
                    crop_x = crop_x - boundary_x
                    crop_y = crop_y - boundary_y
                    # set new crop acording to affine size
                    crop_y_stop = crop_y + affine_size[0]
                    crop_x_stop = crop_x + affine_size[1]
                    img_crops = seq_data['images'][:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                    img_max = seq_data['images'].max()
                    seg_crops = seq_data['segs'][:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                    # center crop for affine transformations:
                    y_center_crop = boundary_y
                    y_center_crop_stop = y_center_crop + self.sub_seq_size[0]
                    x_center_crop = boundary_x
                    x_center_crop_stop = x_center_crop + self.sub_seq_size[1]
                    if self.output_tra:
                        tra_crops = seq_data['tra'][:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                    else:
                        tra_crops = None

                all_fnames = []
                # apply augmentations
                for t, file_idx in enumerate(filename_idx):
                    all_times = [time.time()]
                    filename = seq_data['metadata']['filelist'][file_idx][0]
                    img_crop = img_crops[file_idx].copy()
                    seg_crop = seg_crops[file_idx].copy()
                    if self.output_tra:
                        tra_crop = tra_crops[file_idx].copy()
                    full_seg = seq_data['full_seg'][file_idx]
                    if self.randomize:
                        # contrast factor between [0.5, 1.5]
                        random_constrast_factor = np.random.rand() + 0.5
                        # random brightness delta plus/minus 10% of maximum value
                        random_brightness_delta = (np.random.rand() - 0.5) * 0.2 * img_max
                        img_crop = self._adjust_contrast_(img_crop, random_constrast_factor)
                        img_crop = self._adjust_brightness_(img_crop, random_brightness_delta)

                    if not boundary_flag:
                        if self.elastic_augmentation:
                            trans_img = self._get_transformed_image_(img_crop, affine_matrix, indices)
                            img_crop = trans_img
                            if np.any(np.isnan(img_crop)):
                                raise ValueError('NaN in image {} from sequence: {}'.format(file_idx, sequence_folder))
                            if np.any(np.isinf(img_crop)):
                                raise ValueError('Inf in image {} from sequence: {}'.format(file_idx, sequence_folder))
                            # center crop for image
                            img_crop = img_crop[y_center_crop:y_center_crop_stop, x_center_crop:x_center_crop_stop]
                            # img_crop = img_crop[crop_y:crop_y_stop, crop_x:crop_x_stop]
                            if not np.equal(seg_crop, -1).all():
                                seg_not_valid = np.equal(seg_crop, -1)
                                labeled_gt = seg_crop
                                labeled_gt[:, 0] = 0
                                labeled_gt[:, -1] = 0
                                labeled_gt[-1, :] = 0
                                labeled_gt[0, :] = 0
                                trans_seg = self._get_transformed_image_(labeled_gt.astype(np.float32), affine_matrix,
                                                                         indices, seg=True)
                                trans_not_valid = self._get_transformed_image_(seg_not_valid.astype(np.float32),
                                                                               affine_matrix,
                                                                               indices, seg=True)
                                trans_seg_fix = self._fix_transformed_segmentation(trans_seg)
                                trans_not_valid = np.logical_or(np.greater(trans_not_valid, 0.5), np.equal(trans_seg, -1))
                                seg_crop = trans_seg_fix
                                seg_crop[trans_not_valid] = -1
                                if np.any(np.isnan(seg_crop)):
                                    raise ValueError('NaN in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                                if np.any(np.isinf(seg_crop)):
                                    raise ValueError('Inf in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                                # centr crop for seg image
                            seg_crop = seg_crop[y_center_crop:y_center_crop_stop, x_center_crop:x_center_crop_stop]

                            if self.output_tra:
                                if not np.equal(tra_crop, -1).all():
                                    tra_crop = self._get_transformed_image_(tra_crop.astype(np.float32), affine_matrix,
                                                                            indices, seg=True)

                                if np.any(np.isnan(tra_crop)):
                                    raise ValueError('NaN in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                                if np.any(np.isinf(tra_crop)):
                                    raise ValueError('Inf in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                                # center crop tra image
                                tra_crop = tra_crop[y_center_crop:y_center_crop_stop, x_center_crop:x_center_crop_stop]
                        else:
                            seg_crop = self._fix_transformed_segmentation(seg_crop)
                            seg_crop = seg_crop[y_center_crop:y_center_crop_stop, x_center_crop:x_center_crop_stop]
                            img_crop = img_crop[y_center_crop:y_center_crop_stop, x_center_crop:x_center_crop_stop]
                            if self.output_tra:
                                tra_crop = tra_crop[y_center_crop:y_center_crop_stop, x_center_crop:x_center_crop_stop]
                    else:
                        seg_crop = self._fix_transformed_segmentation(seg_crop)

                    if flip[0]:
                        img_crop = cv2.flip(img_crop, 0)
                        seg_crop = cv2.flip(seg_crop, 0)
                        if self.output_tra:
                            tra_crop = cv2.flip(tra_crop, 0)
                    if flip[1]:
                        img_crop = cv2.flip(img_crop, 1)
                        seg_crop = cv2.flip(seg_crop, 1)
                        if self.output_tra:
                            tra_crop = cv2.flip(tra_crop, 1)
                    if rotate > 0:
                        img_crop = np.rot90(img_crop, rotate)
                        seg_crop = np.rot90(seg_crop, rotate)
                        if self.output_tra:
                            tra_crop = np.rot90(tra_crop, rotate)

                    if self.output_tra:
                        all_times.append(time.time())
                        processed_tra[t] = tra_crop
                    is_last_frame = 1. if (t + 1) < len(filename_idx) else 0.
                    if self.num_threads == 1:
                        try:

                            while q_stat().numpy() > 0.9:
                                if self.coord.should_stop():
                                    return
                                time.sleep(1)

                            if self.output_tra:
                                q.enqueue(
                                    [img_crop, seg_crop, max(0, full_seg), is_last_frame, filename, tra_crop])
                            else:
                                q.enqueue([img_crop, seg_crop, max(0, full_seg), is_last_frame, filename])

                        except tf.errors.CancelledError:
                            pass
                    else:
                        is_last[t] = is_last_frame
                        processed_image[t] = img_crop
                        processed_seg[t] = seg_crop
                        full_seg_seq[t] = max(0, full_seg)
                        all_fnames.append(filename)
                        if self.coord.should_stop():
                            return
                if self.num_threads == 1:
                    continue

                try:

                    while q_stat().numpy() > 0.9:
                        if self.coord.should_stop():
                            return
                        time.sleep(1)

                    if self.output_tra:
                        q.enqueue_many(
                            [processed_image[:t + 1], processed_seg[:t + 1], full_seg_seq[:t + 1], is_last[:t + 1],
                             all_fnames,
                             processed_tra[:t + 1]])
                    else:
                        q.enqueue_many(
                            [processed_image[:t + 1], processed_seg[:t + 1], full_seg_seq[:t + 1], is_last[:t + 1],
                             all_fnames])

                except tf.errors.CancelledError:
                    pass

        except tf.errors.CancelledError:
            pass

        except Exception as err:
            print('ERROR FROM DATA PROCESS')
            self.coord.request_stop(err)
            raise err

    def _create_queues(self):
        def normed_size(_q):
            @tf.function
            def q_stat():
                return tf.cast(_q.size(), tf.float32) / self.queue_capacity

            return q_stat

        with tf.name_scope('DataHandler'):
            dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.string]
            shapes = [self.sub_seq_size, self.sub_seq_size, (), (), ()]
            if self.output_tra:
                dtypes += [tf.float32]
                shapes += [self.sub_seq_size]
            q_list = []

            q_stat_list = []
            for b in range(self.batch_size):
                q = tf.queue.FIFOQueue(self.queue_capacity, dtypes=dtypes, shapes=shapes, name='data_q_{}'.format(b))
                q_list.append(q)
                q_stat_list.append(normed_size(q))

        return q_list, q_stat_list

    def _batch_queues_(self):
        with tf.name_scope('DataHandler'):
            img_list = []
            seg_list = []
            tra_list = []
            full_seg_list = []
            is_last_list = []
            # fname_list = []
            for q in self.q_list:
                if self.output_tra:
                    img, seg, full_seg, is_last, fnames, tra = q.dequeue_many(self.unroll_len)
                    tra_list.append(tra)
                else:
                    img, seg, full_seg, is_last, fnames = q.dequeue_many(self.unroll_len)
                img_list.append(img)
                seg_list.append(seg)
                full_seg_list.append(full_seg)
                is_last_list.append(is_last[-1])
                # fname_list.append(fnames)

            image_batch = tf.stack(img_list, axis=0)
            seg_batch = tf.stack(seg_list, axis=0)
            # fnames_batch = tf.stack(fname_list, axis=0)
            full_seg_batch = tf.stack(full_seg_list, axis=0)
            is_last_batch = tf.stack(is_last_list, axis=0)
            tra_batch = tf.stack(tra_list, axis=0) if self.output_tra else None

            if self.data_format == 'NHWC':
                image_batch = tf.expand_dims(image_batch, 4)
                seg_batch = tf.expand_dims(seg_batch, 4)
                if self.output_tra:
                    tra_batch = tf.expand_dims(tra_batch, 4)
            elif self.data_format == 'NCHW':
                image_batch = tf.expand_dims(image_batch, 2)
                seg_batch = tf.expand_dims(seg_batch, 2)
                if self.output_tra:
                    tra_batch = tf.expand_dims(tra_batch, 2)
            else:
                raise ValueError()

        if self.output_tra:
            return image_batch, seg_batch, full_seg_batch, is_last_batch, tra_batch

        return image_batch, seg_batch, full_seg_batch, is_last_batch

    def _create_sequence_queue(self):
        sequence_queue = queue.Queue(maxsize=len(self.sequence_folder_list))
        for sequence in self.sequence_folder_list:
            sequence_queue.put(sequence)
        return sequence_queue

    def start_queues(self, coord=tf.train.Coordinator(), debug=False):
        self._read_sequence_to_ram_()
        threads = []
        self.coord = coord
        for q, q_stat in zip(self.q_list, self.q_stat_list):
            if debug:
                self._load_and_enqueue(q, q_stat)
            for _ in range(self.num_threads):
                t = threading.Thread(target=self._load_and_enqueue, args=(q, q_stat))
                t.daemon = True
                t.start()
                threads.append(t)
                self.coord.register_thread(t)

        t = threading.Thread(target=self._monitor_queues_)
        t.daemon = True
        t.start()
        self.coord.register_thread(t)
        threads.append(t)
        return threads

    def _monitor_queues_(self):
        while not self.coord.should_stop():
            time.sleep(1)
        for q in self.q_list:
            q.close(cancel_pending_enqueues=True)

    def get_batch(self):
        return self._batch_queues_()


class LSCRAMReader2D(object):
    def __init__(self, sequence_folder_list, image_crop_size=(128, 128), unroll_len=7, deal_with_end=0, batch_size=4,
                 queue_capacity=32, num_threads=3,
                 data_format='NCHW', randomize=True, return_dist=False, keep_sample=1, elastic_augmentation=True):
        self.coord = None
        self.unroll_len = unroll_len
        self.sequence_data = {}
        self.sequence_folder_list = sequence_folder_list
        self.elastic_augmentation = elastic_augmentation
        self.sub_seq_size = image_crop_size
        self.dist_sub_seq_size = (2,) + image_crop_size
        self.deal_with_end = deal_with_end
        self.batch_size = batch_size
        self.queue_capacity = queue_capacity
        self.data_format = data_format
        self.randomize = randomize
        self.return_dist = return_dist
        self.num_threads = num_threads
        self.keep_sample = keep_sample

        self.q_list, self.q_stat_list = self._create_queues()
        np.random.seed(1)

    @classmethod
    def unit_test(cls):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        root_dir = '/media/rrtammyfs/labDatabase/LeafSegmentation/CVPPP2014_LSC_training_data'

        folder_list = [os.path.join(root_dir, 'A1'),
                       os.path.join(root_dir, 'A2'),
                       os.path.join(root_dir, 'A3')]
        image_crop_size = (128, 128)
        unroll_len = 7
        deal_with_end = 0
        batch_size = 4
        queue_capacity = 250
        num_threads = 2
        data_format = 'NCHW'
        randomize = True
        return_dist = False
        keep_sample = 1
        elastic_augmentation = True
        data = cls(folder_list, image_crop_size, unroll_len, deal_with_end, batch_size, queue_capacity,
                   num_threads, data_format, randomize, return_dist, keep_sample, elastic_augmentation)

        debug = True
        data.start_queues(debug=debug)
        for i in range(100):
            image_batch, seg_batch, full_seg, is_last, fnames = data.get_batch()
            utils.log_print(image_batch.shape, seg_batch.shape, is_last.shape)

    def _read_sequence_to_ram_(self):
        for sequence_folder in self.sequence_folder_list:
            train_set = True
            seq = None
            if isinstance(sequence_folder, tuple):
                if len(sequence_folder) == 2:
                    sequence_folder, seq = sequence_folder
                elif len(sequence_folder) == 3:
                    sequence_folder, seq, train_set = sequence_folder

            utils.log_print('Reading Sequence {}: {}'.format(sequence_folder, seq))
            filename_list = glob.glob(os.path.join(sequence_folder, 'plant*_label.png'))
            metadata = {'filelist': filename_list, 'max_value': 0.}
            all_images = []  # = np.zeros((len(filename_list), img_size[0], img_size[1]))
            all_seg = []  # np.zeros((len(filename_list), img_size[0], img_size[1]))
            all_full_seg = []  # np.zeros((len(filename_list)))
            keep_rate = 1
            original_size = 0
            downampled_size = 0
            for t, filename in enumerate(filename_list):
                filename = os.path.basename(filename)
                rgb_filename = filename.replace('label', 'rgb')
                img = cv2.cvtColor(cv2.imread(os.path.join(sequence_folder, rgb_filename)),
                                   cv2.COLOR_BGRA2RGB)
                if img is None:
                    raise ValueError('Could not load image: {}'.format(os.path.join(sequence_folder, rgb_filename)))
                if 'shape' not in metadata.keys():
                    metadata['shape'] = img.shape
                img = img.astype(np.float32)
                img = (img - img.mean()) / (img.std())
                full_seg = 1
                if full_seg == 1:
                    original_size += 1

                keep_seg = (np.random.rand() < keep_rate) and train_set

                full_seg = full_seg if keep_seg else 0
                if full_seg == 1:
                    downampled_size += 1

                if filename is None or not keep_seg:
                    seg = np.ones(img.shape[:2]) * (-1)
                elif not full_seg:
                    seg = cv2.imread(os.path.join(sequence_folder, filename), -1)
                    if seg is None:
                        seg = np.ones(img.shape[:2]) * (-1)
                        full_seg = -1

                    else:
                        seg = seg.astype(np.float32)
                    seg[seg == 0] = -1
                else:
                    seg = cv2.imread(os.path.join(sequence_folder, filename), cv2.IMREAD_GRAYSCALE)
                all_images.append(img)
                metadata['max_value'] = np.maximum(metadata['max_value'], img.max())
                all_seg.append(seg)
                all_full_seg.append(full_seg)
            if keep_rate < 1:
                print('Downsampling Training Segmentaiont with rate:{}. Original set size: {}. '
                      'Downsampled set size: {}'.format(keep_rate, original_size, downampled_size))

            self.sequence_data[sequence_folder] = {'images': all_images, 'segs': all_seg,
                                                   'full_seg': all_full_seg, 'metadata': metadata}

    def _read_sequence_data(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        if isinstance(sequence_folder, tuple):
            sequence_folder = sequence_folder[0]
        return self.sequence_data[sequence_folder], sequence_folder

    def _read_sequence_metadata(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        if isinstance(sequence_folder, tuple):
            sequence_folder = sequence_folder[0]
        with open(os.path.join(sequence_folder, 'metadata.pickle'), 'rb') as fobj:
            metadata = pickle.load(fobj)
        return metadata

    @staticmethod
    def _get_elastic_affine_matrix_(shape_size, alpha_affine):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
         .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
              Convolutional Neural Networks applied to Visual Document Analysis", in
              Proc. of the International Conference on Document Analysis and
              Recognition, 2003.

          Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         """
        random_state = np.random.RandomState(None)

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        affine_matrix = cv2.getAffineTransform(pts1, pts2)

        return affine_matrix, random_state

    @staticmethod
    def _get_transformed_image_(image, affine_matrix, indices, seg=False):

        shape = image.shape[:2]
        if seg:
            trans_img = cv2.warpAffine(image, affine_matrix, shape[::-1], borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=-1, flags=cv2.INTER_NEAREST)
            trans_coord = map_coordinates(trans_img, indices, order=0, mode='constant', cval=-1).reshape(shape)
        else:
            trans_img = cv2.warpAffine(image, affine_matrix, shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
            trans_coord = map_coordinates(trans_img, indices, order=1, mode='reflect').reshape(shape)

        return trans_coord

    @staticmethod
    def _get_indices4elastic_transform(shape, alpha, sigma, random_state):
        dxr = random_state.rand(*shape)
        dx = gaussian_filter((dxr * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        return indices

    @staticmethod
    def _fix_transformed_segmentation(trans_seg):
        # separate close blobs with the edge of transformation
        # edge is pixels that get fractional value after transofrmation
        strel = np.zeros((3, 3))
        trans_seg = np.round(trans_seg)
        # region_props = skimage.measure.regionprops(trans_seg)
        # errosion = grey_erosion(trans_seg, np.zeros(3, 3, 3))
        dilation = grey_dilation(trans_seg.astype(np.int32), structure=strel.astype(np.int8))
        bw = np.minimum(trans_seg, 1)
        bw[np.logical_and(np.not_equal(trans_seg, dilation), np.greater(dilation, 0))] = 2

        return bw

    @staticmethod
    def _gt2dist_(gt_image):
        gt_fg = gt_image == 1
        _, labeled_gt = cv2.connectedComponents(gt_fg.astype(np.uint8))
        im_shape = gt_image.shape
        dist_1 = np.ones_like(gt_image) * (im_shape[0] + im_shape[1]) + 2.
        dist_2 = dist_1 + 1.

        for label in np.unique(labeled_gt):
            if label == 0:
                continue
            bw = np.equal(labeled_gt, label).astype(np.float32)
            bw_erode = cv2.erode(bw, np.ones((3, 3)))
            edge = np.logical_and(np.logical_not(bw_erode), bw)

            dist = distance_transform_edt(np.logical_not(edge))
            is_first_dist = np.less(dist, dist_1)
            dist_2[is_first_dist] = dist_1[is_first_dist]
            is_second_dist = np.logical_and(np.less(dist, dist_2), np.logical_not(is_first_dist))

            dist_1[is_first_dist] = dist[is_first_dist]
            dist_2[is_second_dist] = dist[is_second_dist]
        out = np.stack((dist_1, dist_2), 0)

        return out, (dist_1, dist_2)

    @staticmethod
    def _adjust_brightness_(image, delta):
        """
        Args:
        image (numpy.ndarray)
        delta
        """

        out_img = image + delta
        return out_img

    @staticmethod
    def _adjust_contrast_(image, factor):
        """
        Args:
        image (numpy.ndarray)
        factor
        """

        img_mean = image.mean()
        out_img = (image - img_mean) * factor + img_mean
        return out_img

    def _load_and_enqueue(self, q, q_stat):

        unroll_len = self.unroll_len
        try:
            while not self.coord.should_stop():
                seq_data, sequence_folder = self._read_sequence_data()
                img_size = seq_data['metadata']['shape']

                if img_size[0] - self.sub_seq_size[0] > 0:
                    crop_y = np.random.randint(0, img_size[0] - self.sub_seq_size[0]) if self.randomize else 0
                else:
                    crop_y = 0
                if img_size[1] - self.sub_seq_size[1] > 0:
                    crop_x = np.random.randint(0, img_size[1] - self.sub_seq_size[1]) if self.randomize else 0
                else:
                    crop_x = 0

                flip = np.random.randint(0, 2, 2) if self.randomize else [0, 0]
                rotate = np.random.randint(0, 4) if self.randomize else 0
                if self.elastic_augmentation:
                    affine_matrix, random_state = self._get_elastic_affine_matrix_(self.sub_seq_size,
                                                                                   self.sub_seq_size[1] * 0.08)
                    indices = self._get_indices4elastic_transform(self.sub_seq_size, self.sub_seq_size[1] * 2,
                                                                  self.sub_seq_size[1] * 0.15,
                                                                  random_state)
                else:
                    affine_matrix = indices = None

                filename_idx = list(range(len(seq_data['metadata']['filelist'])))
                seq_len = len(filename_idx)
                remainder = seq_len % unroll_len

                if remainder:
                    if self.deal_with_end == 0:
                        filename_idx = filename_idx[:-remainder]
                    elif self.deal_with_end == 1:
                        filename_idx += filename_idx[-2:-unroll_len + remainder - 2:-1]
                    elif self.deal_with_end == 2:
                        filename_idx += filename_idx[-1:] * (unroll_len - remainder)
                crop_y_stop = crop_y + self.sub_seq_size[0]
                crop_x_stop = crop_x + self.sub_seq_size[1]
                file_idx = random.choice(filename_idx)
                img_crops = seq_data['images'][file_idx][crop_y:crop_y_stop, crop_x:crop_x_stop]
                img_max = seq_data['images'][file_idx].max()

                seg_crops = seq_data['segs'][file_idx][crop_y:crop_y_stop, crop_x:crop_x_stop]
                processed_image = []
                processed_seg = []
                full_seg_seq = []
                processed_dist = []
                is_last = []
                all_fnames = []
                all_times = [time.time()]
                filename = seq_data['metadata']['filelist'][file_idx]
                img_crop = img_crops.copy()
                seg_crop = seg_crops.copy()
                full_seg = seq_data['full_seg'][file_idx]
                if self.randomize:
                    # contrast factor between [0.5, 1.5]
                    random_constrast_factor = np.random.rand() + 0.5
                    # random brightness delta plus/minus 10% of maximum value
                    random_brightness_delta = (np.random.rand() - 0.5) * 0.2 * img_max
                    img_crop = self._adjust_contrast_(img_crop, random_constrast_factor)
                    img_crop = self._adjust_brightness_(img_crop, random_brightness_delta)

                if self.elastic_augmentation:
                    trans_img = self._get_transformed_image_(img_crop, affine_matrix, indices)
                    img_crop = trans_img
                    if np.any(np.isnan(img_crop)):
                        raise ValueError('NaN in image {} from sequence: {}'.format(file_idx, sequence_folder))
                    if np.any(np.isinf(img_crop)):
                        raise ValueError('Inf in image {} from sequence: {}'.format(file_idx, sequence_folder))
                    if not np.equal(seg_crop, -1).all():
                        seg_not_valid = np.equal(seg_crop, -1)
                        labeled_gt = seg_crop
                        labeled_gt[:, 0] = 0
                        labeled_gt[:, -1] = 0
                        labeled_gt[-1, :] = 0
                        labeled_gt[0, :] = 0
                        trans_seg = self._get_transformed_image_(labeled_gt.astype(np.float32), affine_matrix,
                                                                 indices, seg=True)
                        trans_not_valid = self._get_transformed_image_(seg_not_valid.astype(np.float32),
                                                                       affine_matrix,
                                                                       indices, seg=True)
                        trans_seg_fix = self._fix_transformed_segmentation(trans_seg)
                        trans_not_valid = np.logical_or(np.greater(trans_not_valid, 0.5), np.equal(trans_seg, -1))
                        seg_crop = trans_seg_fix
                        seg_crop[trans_not_valid] = -1
                        if np.any(np.isnan(seg_crop)):
                            raise ValueError('NaN in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                        if np.any(np.isinf(seg_crop)):
                            raise ValueError('Inf in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                else:
                    seg_crop = self._fix_transformed_segmentation(seg_crop)
                if flip[0]:
                    img_crop = cv2.flip(img_crop, 0)
                    seg_crop = cv2.flip(seg_crop, 0)
                if flip[1]:
                    img_crop = cv2.flip(img_crop, 1)
                    seg_crop = cv2.flip(seg_crop, 1)
                if rotate > 0:
                    img_crop = np.rot90(img_crop, rotate)
                    seg_crop = np.rot90(seg_crop, rotate)
                if self.return_dist:
                    if full_seg == -1:
                        dist_crop = np.zeros(self.dist_sub_seq_size)
                    else:
                        dist_crop, _ = self._gt2dist_(seg_crop)
                    all_times.append(time.time())
                    processed_dist.append(dist_crop)
                is_last_frame = 1.
                is_last.append(is_last_frame)
                processed_image.append(img_crop)
                processed_seg.append(seg_crop)
                all_fnames.append(filename)
                full_seg_seq.append(max(0, full_seg))
                if self.coord.should_stop():
                    return

                try:
                    while q_stat().numpy() > 0.9:
                        if self.coord.should_stop():
                            return
                        time.sleep(1)

                    if self.return_dist:
                        q.enqueue_many([processed_image, processed_seg, full_seg_seq, is_last, all_fnames,
                                        processed_dist])
                    else:
                        q.enqueue_many([processed_image, processed_seg, full_seg_seq, is_last, all_fnames])

                except tf.errors.CancelledError:
                    pass

        except tf.errors.CancelledError:
            pass

        except Exception as err:
            print('ERROR FROM DATA PROCESS')
            self.coord.request_stop(err)
            raise err

    def _create_queues(self):
        def normed_size(_q):
            @tf.function
            def q_stat():
                return tf.cast(_q.size(), tf.float32) / self.queue_capacity

            return q_stat

        with tf.name_scope('DataHandler'):
            dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.string]
            shapes = [self.sub_seq_size, self.sub_seq_size, (), (), ()]
            if self.return_dist:
                dtypes += [tf.float32]
                shapes += [self.dist_sub_seq_size]
            q_list = []

            q_stat_list = []
            for b in range(self.batch_size):
                q = tf.queue.FIFOQueue(self.queue_capacity, dtypes=dtypes, shapes=shapes, name='data_q_{}'.format(b))
                q_list.append(q)
                q_stat_list.append(normed_size(q))

        return q_list, q_stat_list

    def _batch_queues_(self):
        with tf.name_scope('DataHandler'):
            img_list = []
            seg_list = []
            dist_list = []
            full_seg_list = []
            is_last_list = []
            fname_list = []
            for q in self.q_list:
                if self.return_dist:
                    img, seg, full_seg, is_last, fnames, dist = q.dequeue_many(self.unroll_len)
                    dist_list.append(dist)
                else:
                    img, seg, full_seg, is_last, fnames = q.dequeue_many(self.unroll_len)
                img_list.append(img)
                seg_list.append(seg)
                full_seg_list.append(full_seg)
                is_last_list.append(is_last[-1])
                fname_list.append(fnames)

            image_batch = tf.stack(img_list, axis=0)
            seg_batch = tf.stack(seg_list, axis=0)
            fnames_batch = tf.stack(fname_list, axis=0)
            full_seg_batch = tf.stack(full_seg_list, axis=0)
            is_last_batch = tf.stack(is_last_list, axis=0)
            dist_batch = tf.stack(dist_list, axis=0) if self.return_dist else None

            if self.data_format == 'NHWC':
                image_batch = tf.expand_dims(image_batch, 4)
                seg_batch = tf.expand_dims(seg_batch, 4)
            elif self.data_format == 'NCHW':
                image_batch = tf.expand_dims(image_batch, 2)
                seg_batch = tf.expand_dims(seg_batch, 2)
            else:
                raise ValueError()

        if self.return_dist:
            return image_batch, seg_batch, full_seg_batch, is_last_batch, dist_batch, fnames_batch

        return image_batch, seg_batch, full_seg_batch, is_last_batch, fnames_batch

    def _create_sequence_queue(self):
        sequence_queue = queue.Queue(maxsize=len(self.sequence_folder_list))
        for sequence in self.sequence_folder_list:
            sequence_queue.put(sequence)
        return sequence_queue

    def start_queues(self, coord=tf.train.Coordinator(), debug=False):
        self._read_sequence_to_ram_()
        threads = []
        self.coord = coord
        for q, q_stat in zip(self.q_list, self.q_stat_list):
            if debug:
                self._load_and_enqueue(q, q_stat)
            for _ in range(self.num_threads):
                t = threading.Thread(target=self._load_and_enqueue, args=(q, q_stat))
                t.daemon = True
                t.start()
                threads.append(t)
                self.coord.register_thread(t)

        t = threading.Thread(target=self._monitor_queues_)
        t.daemon = True
        t.start()
        self.coord.register_thread(t)
        threads.append(t)
        return threads

    def _monitor_queues_(self):
        while not self.coord.should_stop():
            time.sleep(1)
        for q in self.q_list:
            q.close(cancel_pending_enqueues=True)

    def get_batch(self):
        return self._batch_queues_()


class CTCSegReaderSequence3D(object):
    def __init__(self, sequence_folder_list, image_crop_size=(32, 128, 128), unroll_len=7, deal_with_end=0,
                 batch_size=4, queue_capacity=32, num_threads=3, data_format='NCDHW', randomize=True, return_dist=False,
                 keep_sample=1, elastic_augmentation=False, switch_to_local_db=False, load_to_ram=False,
                 local_db_replace=('/persistent', '/data3d')):
        if not isinstance(image_crop_size, tuple):
            image_crop_size = tuple(image_crop_size)
        self.unroll_len = unroll_len
        self.sequence_data = {}
        self.sequence_folder_list = sequence_folder_list
        self.augmentation = elastic_augmentation
        self.sub_seq_size = image_crop_size
        self.dist_sub_seq_size = (2,) + image_crop_size
        self.deal_with_end = deal_with_end
        self.batch_size = batch_size
        self.queue_capacity = queue_capacity
        self.data_format = data_format
        self.randomize = randomize
        self.return_dist = return_dist
        self.num_threads = num_threads
        self.keep_sample = keep_sample
        self.switch_to_local_db = switch_to_local_db
        self.local_db_replace = local_db_replace
        self.load_to_ram = load_to_ram
        self.ram_data = {}

        self.q_list, self.q_stat_list = self._create_queues()
        self.coord = None

    @classmethod
    def unit_test(cls):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/'
        dataset = 'Fluo-C3DL-MDA231'
        sequence = '01'
        train_data = True
        image_crop_size = (16, 80, 80)
        unroll_len = 3
        num_threads = 3
        batch_size = 4
        debug = True
        queue_capacity = 200
        sequence_folder_list = [(os.path.join(root_dir, dataset), sequence, train_data)]
        data = cls(sequence_folder_list, batch_size=batch_size, image_crop_size=image_crop_size, unroll_len=unroll_len,
                   elastic_augmentation=True, num_threads=num_threads, queue_capacity=queue_capacity)
        coord = tf.train.Coordinator()
        data.start_queues(coord, debug=debug)
        time.sleep(30)
        for i in range(100):
            time.sleep(0.1)
            start = time.time()
            image_batch_list, seg_batch_list, full_seg_batch_list, is_last_batch = data.get_batch()
            print([qs().numpy() for qs in data.q_stat_list])
            print('Time: {}'.format(time.time() - start))
            # print([q.size().numpy() for q in data.q_list])
            print(image_batch_list[0].shape)

    def _read_sequence_to_ram_(self):

        for sequence_folder in self.sequence_folder_list:
            train_set = True
            seq = '01'
            sequence_folder_orig = sequence_folder
            if isinstance(sequence_folder, tuple):
                sequence_folder, seq, train_set = sequence_folder

            utils.log_print('Reading Sequence {}'.format(sequence_folder))
            with open(os.path.join(sequence_folder, 'metadata_{}.pickle'.format(seq)), 'rb') as fobj:
                metadata = pickle.load(fobj)
            filename_list = metadata['filelist']
            original_size = 0
            keep_inds = []
            for t, f in enumerate(filename_list):
                if f[1] is not None and f[2] is True:
                    original_size += 1
                    keep_inds.append(t)
                elif isinstance(f[1], list):
                    for z, (slice_f, valid_f) in enumerate(zip(f[1], f[2])):
                        if slice_f is not None and valid_f is True:
                            original_size += 1
                            keep_inds.append(t)
                            break
            keep_inds = np.array(keep_inds)

            keep_rate = self.keep_sample if train_set else 0
            keep_num = np.round(keep_rate * original_size).astype(np.int16)
            keep_vec = np.zeros(len(filename_list))
            if keep_num > 0 and len(keep_inds) > 0:
                keep_inds = np.random.choice(keep_inds, keep_num, replace=False)
                keep_vec[keep_inds] = 1
            downampled_size = keep_num

            for t, (filename, keep_seg) in enumerate(zip(filename_list, keep_vec)):

                if not keep_seg:
                    filename[1] = None
                    filename[2] = None

            if keep_rate < 1:
                print('Downsampling Training Segmentaiont with rate:{}. Original set size: {}. '
                      'Downsampled set size: {}'.format(keep_rate, original_size, downampled_size))

            self.sequence_data[sequence_folder_orig] = {'metadata': metadata}

    def _read_sequence_data(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        # if isinstance(sequence_folder, tuple):
        #     sequence_folder = sequence_folder[0]
        return self.sequence_data[sequence_folder], sequence_folder

    def _read_sequence_metadata(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        if isinstance(sequence_folder, tuple):
            sequence_folder = sequence_folder[0]
        with open(os.path.join(sequence_folder, 'metadata.pickle'), 'rb') as fobj:
            metadata = pickle.load(fobj)
        return metadata

    @staticmethod
    def _points2affine3d_(pts1, pts2):
        """
         pts need to be a set of 4 point of shape 4x3 for zyx
        :param pts1:
        :param pts2:
        :return:
        """
        x_mat = np.zeros((12, 12))
        for p_ind, pt in enumerate(pts1):
            pt1 = np.concatenate((pt, np.array([1])), axis=0)
            x_mat[p_ind * 3] = np.concatenate((pt1, np.zeros(8)), axis=0)
            x_mat[p_ind * 3 + 1] = np.concatenate((np.zeros(4), pt1, np.zeros(4)), axis=0)
            x_mat[p_ind * 3 + 2] = np.concatenate((np.zeros(8), pt1), axis=0)

        inv_x = np.linalg.inv(x_mat.astype(np.float32))
        pts2 = np.reshape(pts2, -1)

        a = np.dot(inv_x, pts2)
        a_mat = np.concatenate((a, np.array([0, 0, 0, 1])), axis=0)
        a_mat = np.reshape(a_mat, (4, 4))
        return a_mat

    def _get_elastic_affine_matrix_(self, shape_size, alpha_affine):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
         .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
              Convolutional Neural Networks applied to Visual Document Analysis", in
              Proc. of the International Conference on Document Analysis and
              Recognition, 2003.

          Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         """
        random_state = np.random.RandomState(None)

        # Random affine
        depth, height, width = shape_size
        pts1 = np.float32([[0, 0, 0], [0, height, width], [depth, height, 0], [depth, 0, width]])
        pts2 = pts1 + random_state.uniform(size=pts1.shape).astype(np.float32) * (alpha_affine * np.array(shape_size))
        affine_matrix = self._points2affine3d_(pts1, pts2)

        return affine_matrix, random_state

    @staticmethod
    def _get_transformed_image_(image, affine_matrix, indices, seg=False):

        shape = image.shape
        try:
            if seg:
                trans_img = affine_transform(image, affine_matrix, order=0, mode='constant', cval=-1)
                trans_coord = map_coordinates(trans_img, indices, order=0, mode='constant', cval=-1).reshape(shape)
            else:
                trans_img = affine_transform(image, affine_matrix, order=1, mode='reflect')
                trans_coord = map_coordinates(trans_img, indices, order=1, mode='reflect').reshape(shape)
        except RuntimeError:
            a = affine_matrix[:3, :3]
            b = affine_matrix[:3, 3]
            if seg:
                trans_img = affine_transform(image, a, offset=b, order=0, mode='constant', cval=-1)
                trans_coord = map_coordinates(trans_img, indices, order=0, mode='constant', cval=-1).reshape(shape)
            else:
                trans_img = affine_transform(image, a, offset=b, order=1, mode='reflect')
                trans_coord = map_coordinates(trans_img, indices, order=1, mode='reflect').reshape(shape)
        return trans_coord

    @staticmethod
    def _get_indices4elastic_transform(shape, alpha, sigma, random_state):

        dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha[0]
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha[1]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha[2]

        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(z + dz, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        return indices

    @staticmethod
    def _fix_transformed_segmentation(trans_seg, edge_z):
        # separate close blobs with the edge of transformation
        # edge is pixels that get fractional value after transofrmation
        strel = np.zeros((3, 3, 3)) if edge_z else np.zeros((1, 3, 3))
        trans_seg = np.round(trans_seg)
        # region_props = skimage.measure.regionprops(trans_seg)
        # errosion = grey_erosion(trans_seg, np.zeros(3, 3, 3))
        dilation = grey_dilation(trans_seg.astype(np.int32), structure=strel.astype(np.int8))
        bw = np.minimum(trans_seg, 1)
        bw[np.logical_and(np.not_equal(trans_seg, dilation), np.greater(dilation, 0))] = 2
        return bw

    @staticmethod
    def _label_class_segmentation(trans_seg, labels):
        # separate close blobs with the edge of transformation
        # edge is pixels that get fractional value after transofrmation
        trans_seg = np.round(trans_seg)
        bw = np.zeros_like(trans_seg)
        if len(np.unique(trans_seg)) == 0:
            return np.minimum(trans_seg, 1)
        for this_label in labels:
            if this_label == 0 or (this_label not in trans_seg):
                continue

            label_bw = np.equal(trans_seg, this_label)
            # cc = cv2.connectedComponentsWithStats(label_bw.astype(np.uint8), 8, cv2.CV_32S)
            # num = cc[0]
            # label_bw_l = cc[1]
            # stats = cc[2]
            # ll = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            # label_bw = np.equal(label_bw_l, ll)
            label_bw = binary_dilation(label_bw.astype(np.uint8), np.ones((3, 3, 3)))
            bw_dilate = cv2.dilate(label_bw.astype(np.uint8), np.ones((3, 3)))
            edge = np.logical_xor(label_bw, bw_dilate)
            bw[label_bw.astype(np.bool)] = 1
            bw[edge] = 2

        return bw

    @staticmethod
    def _gt2dist_(gt_image):
        gt_fg = gt_image == 1
        _, labeled_gt = cv2.connectedComponents(gt_fg.astype(np.uint8))
        im_shape = gt_image.shape
        dist_1 = np.ones_like(gt_image) * (im_shape[0] + im_shape[1]) + 2.
        dist_2 = dist_1 + 1.

        for label in np.unique(labeled_gt):
            if label == 0:
                continue
            bw = np.equal(labeled_gt, label).astype(np.float32)
            bw_erode = cv2.erode(bw, np.ones((3, 3)))
            edge = np.logical_and(np.logical_not(bw_erode), bw)

            dist = distance_transform_edt(np.logical_not(edge))
            is_first_dist = np.less(dist, dist_1)
            dist_2[is_first_dist] = dist_1[is_first_dist]
            is_second_dist = np.logical_and(np.less(dist, dist_2), np.logical_not(is_first_dist))

            dist_1[is_first_dist] = dist[is_first_dist]
            dist_2[is_second_dist] = dist[is_second_dist]
        out = np.stack((dist_1, dist_2), 0)

        return out, (dist_1, dist_2)

    def read_images(self, sequence_folder, file_idx, start_z=None, stop_z=None, ):
        load_to_ram = self.load_to_ram
        sequence_folder_orig = sequence_folder
        if isinstance(sequence_folder, tuple):
            sequence_folder, seq, train_set = sequence_folder
        if self.switch_to_local_db and os.path.exists(os.path.join(self.local_db_replace[1], '.done_data_move')):
            this_dir = sequence_folder.replace(self.local_db_replace[0], self.local_db_replace[1])
        else:
            this_dir = sequence_folder
        metadata = self.sequence_data[sequence_folder_orig]['metadata']
        filename_list = metadata['filelist']
        filename = filename_list[file_idx]
        data_path = os.path.join(this_dir, filename[0])
        if load_to_ram:
            if data_path not in self.ram_data.keys():
                full_img = utils.read_multi_tiff(data_path)
                if full_img is None:
                    print('Could Not Load Image: {}'.format(data_path))
                self.ram_data[data_path] = full_img
            else:
                full_img = self.ram_data[data_path]
            start_z = max(0, start_z)
            stop_z = min(stop_z, full_img.shape[0])
            img = full_img[start_z:stop_z].copy()
        else:
            img = utils.read_multi_tiff(data_path, start_z=start_z, stop_z=stop_z)
        if img is None:
            print('Could Not Load Image: {}'.format(data_path))

        img = img.astype(np.float32)
        img = (img - img.mean()) / (img.std())
        seg = None
        full_seg = 1 if filename[2] is True else 0

        if filename[1] is None:
            seg = np.ones(img.shape[:3]) * (-1)
        elif not full_seg:
            if isinstance(filename[1], str):
                seg_path = os.path.join(this_dir, filename[1])
                if load_to_ram:
                    if seg_path not in self.ram_data.keys():
                        full_seg = utils.read_multi_tiff(seg_path).astype(np.float32)
                        if full_seg is None:
                            print('Could Not Load Image: {}'.format(seg_path))
                        self.ram_data[seg_path] = full_seg
                    else:
                        full_seg = self.ram_data[seg_path]
                    start_z = max(0, start_z)
                    stop_z = min(stop_z, full_seg.shape[0])
                    seg = full_seg[start_z:stop_z].copy()
                else:

                    seg = utils.read_multi_tiff(seg_path, start_z=start_z, stop_z=stop_z)
                    seg = seg.astype(np.float32)

            elif isinstance(filename[1], list):
                seg = np.ones(img.shape[:3]) * (-1)
                for slice_name, valid in zip(filename[1], filename[2]):
                    z = int(slice_name[-7:-4]) - start_z
                    if z < 0 or z >= seg.shape[0]:
                        continue
                    seg_path = os.path.join(this_dir, slice_name)

                    if load_to_ram:
                        if seg_path not in self.ram_data.keys():
                            this_slice = cv2.imread(seg_path, -1).astype(np.float32)
                            if this_slice is None:
                                print('Could Not Load Image: {}'.format(seg_path))
                            self.ram_data[seg_path] = this_slice
                        else:
                            this_slice = self.ram_data[seg_path]
                    else:
                        this_slice = cv2.imread(seg_path, -1).astype(np.float32)

                    if not valid or 'DRO' in sequence_folder:
                        this_slice[this_slice == 0] = -1
                    seg[z] = this_slice
                seg = seg

        else:
            seg_path = os.path.join(this_dir, filename[1])
            if load_to_ram:
                if seg_path not in self.ram_data.keys():
                    full_seg = utils.read_multi_tiff(seg_path).astype(np.float32)
                    if full_seg is None:
                        print('Could Not Load Image: {}'.format(seg_path))
                    self.ram_data[seg_path] = full_seg
                else:
                    full_seg = self.ram_data[seg_path]
                start_z = max(0, start_z)
                stop_z = min(stop_z, full_seg.shape[0])
                seg = full_seg[start_z:stop_z].copy()
            else:

                seg = utils.read_multi_tiff(seg_path, start_z=start_z, stop_z=stop_z)
                seg = seg.astype(np.float32)

        return img, seg, full_seg

    def _load_and_enqueue(self, q, q_stat):

        unroll_len = self.unroll_len
        try:
            while not self.coord.should_stop():
                seq_data, sequence_folder = self._read_sequence_data()
                img_size = seq_data['metadata']['shape']
                random_sub_sample = np.random.randint(1, 4) if self.randomize else 0
                random_reverse = np.random.randint(0, 2) if self.randomize else 0
                if img_size[0] - self.sub_seq_size[0] > 0:
                    crop_z = np.random.randint(0, img_size[0] - self.sub_seq_size[0]) if self.randomize else 0
                else:
                    crop_z = 0
                if img_size[1] - self.sub_seq_size[1] > 0:
                    crop_y = np.random.randint(0, img_size[1] - self.sub_seq_size[1]) if self.randomize else 0
                else:
                    crop_y = 0
                if img_size[2] - self.sub_seq_size[2] > 0:
                    crop_x = np.random.randint(0, img_size[2] - self.sub_seq_size[2]) if self.randomize else 0
                else:
                    crop_x = 0

                flip = np.random.randint(0, 2, 3) if self.randomize else [0, 0, 0]
                rotate = np.random.randint(0, 4) if self.randomize else 0
                if 'SIM' not in sequence_folder[0]:
                    edge_z = False
                else:
                    edge_z = True
                if self.augmentation:
                    affine_alpha = np.array([0.08] * 3)
                    elastic_alpha = np.array(self.sub_seq_size) * 2
                    elastic_sigma = np.array(self.sub_seq_size) * 0.15

                    if not edge_z:
                        affine_alpha[0] = 0.
                        elastic_alpha[0] = 0.

                    affine_matrix, random_state = self._get_elastic_affine_matrix_(self.sub_seq_size, affine_alpha)
                    indices = self._get_indices4elastic_transform(self.sub_seq_size, elastic_alpha,
                                                                  elastic_sigma,
                                                                  random_state)
                else:
                    affine_matrix = indices = None

                filename_idx = list(range(len(seq_data['metadata']['filelist'])))
                if random_reverse:
                    filename_idx.reverse()
                if random_sub_sample:
                    filename_idx = filename_idx[::random_sub_sample]
                seq_len = len(filename_idx)
                remainder = seq_len % unroll_len

                if remainder:
                    if self.deal_with_end == 0:
                        filename_idx = filename_idx[:-remainder]
                    elif self.deal_with_end == 1:
                        filename_idx += filename_idx[-2:-unroll_len + remainder - 2:-1]
                    elif self.deal_with_end == 2:
                        filename_idx += filename_idx[-1:] * (unroll_len - remainder)
                crop_z_stop = crop_z + self.sub_seq_size[0]
                crop_y_stop = crop_y + self.sub_seq_size[1]
                crop_x_stop = crop_x + self.sub_seq_size[2]

                processed_image = []
                processed_seg = []
                full_seg_seq = []
                processed_dist = []
                is_last = []
                img = seg = None
                # start_time = time.time()
                for t, file_idx in enumerate(filename_idx):
                    all_times = [time.time()]
                    if img is not None:
                        del img
                        del seg
                    img, seg, full_seg = self.read_images(sequence_folder, file_idx, start_z=crop_z,
                                                          stop_z=crop_z_stop)
                    img_crop = img[:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                    seg_crop = seg[:, crop_y:crop_y_stop, crop_x:crop_x_stop]

                    if self.augmentation:
                        trans_img = self._get_transformed_image_(img_crop, affine_matrix, indices)
                        img_crop = trans_img
                        if np.any(np.isnan(img_crop)):
                            raise ValueError('NaN in image {} from sequence: {}'.format(file_idx, sequence_folder))
                        if np.any(np.isinf(img_crop)):
                            raise ValueError('Inf in image {} from sequence: {}'.format(file_idx, sequence_folder))
                        if not np.equal(seg_crop, -1).all():

                            # seg_not_valid = np.equal(seg_crop, -1)

                            trans_seg = self._get_transformed_image_(seg_crop.astype(np.float32), affine_matrix,
                                                                     indices, seg=True)
                            # trans_not_valid = self._get_transformed_image_(seg_not_valid.astype(np.float32),
                            #                                                affine_matrix,
                            #                                                indices, seg=True)
                            trans_seg_fix = self._fix_transformed_segmentation(trans_seg, edge_z)
                            # trans_not_valid = np.logical_or(np.greater(trans_not_valid, 0.5), np.equal(trans_seg, -1))
                            seg_crop = trans_seg_fix
                            # seg_crop[trans_not_valid] = -1
                            if np.any(np.isnan(seg_crop)):
                                raise ValueError('NaN in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                            if np.any(np.isinf(seg_crop)):
                                raise ValueError('Inf in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                    else:
                        seg_crop = self._fix_transformed_segmentation(seg_crop, edge_z)
                    if flip[0]:
                        img_crop = np.flip(img_crop, 0)
                        seg_crop = np.flip(seg_crop, 0)
                    if flip[1]:
                        img_crop = np.flip(img_crop, 1)
                        seg_crop = np.flip(seg_crop, 1)
                    if flip[2]:
                        img_crop = np.flip(img_crop, 2)
                        seg_crop = np.flip(seg_crop, 2)
                    if rotate > 0:
                        img_crop = np.rot90(img_crop, rotate, axes=(1, 2))
                        seg_crop = np.rot90(seg_crop, rotate, axes=(1, 2))
                    if self.return_dist:
                        if full_seg == -1:
                            dist_crop = np.zeros(self.dist_sub_seq_size)
                        else:
                            dist_crop, _ = self._gt2dist_(seg_crop)
                        all_times.append(time.time())
                        processed_dist.append(dist_crop)
                    is_last_frame = 1. if (t + 1) < len(filename_idx) else 0.
                    is_last.append(is_last_frame)
                    processed_image.append(img_crop)
                    processed_seg.append(seg_crop)
                    full_seg_seq.append(max(0, full_seg))
                    if self.coord.should_stop():
                        return

                try:
                    while q_stat().numpy() > 0.9:
                        if self.coord.should_stop():
                            return
                        time.sleep(0.5)

                    # print("Thread Time: {}".format(time.time() - start_time))

                    q.enqueue_many([processed_image, processed_seg, full_seg_seq, is_last])

                except tf.errors.CancelledError:
                    pass

        except tf.errors.CancelledError:
            pass

        except Exception as err:
            print('ERROR FROM DATA PROCESS')
            print(str(err))
            self.coord.request_stop(err)
            print('This error:')
            raise err

    def _create_queues(self):

        def normed_size(_q):
            @tf.function
            def q_stat():
                return tf.cast(_q.size(), tf.float32) / self.queue_capacity

            return q_stat

        with tf.device('/cpu:0'):
            with tf.name_scope('DataHandler'):
                dtypes = [tf.float32] * 4
                shapes = [self.sub_seq_size] * 2 + [()] * 2
                q_list = []
                q_stat_list = []
                for b in range(self.batch_size):
                    q = tf.queue.FIFOQueue(self.queue_capacity, dtypes=dtypes, shapes=shapes,
                                           name='data_q_{}'.format(b))
                    q_list.append(q)
                    q_stat_list.append(normed_size(q))

        return q_list, q_stat_list

    @tf.function
    def _batch_queues_(self):
        with tf.name_scope('DataHandler'):
            img_list = []
            seg_list = []
            dist_list = []
            full_seg_list = []
            is_last_list = []
            for q in self.q_list:
                if self.return_dist:
                    img, seg, dist, full_seg, is_last = q.dequeue_many(self.unroll_len)
                    dist_list.append(dist)
                else:
                    img, seg, full_seg, is_last = q.dequeue_many(self.unroll_len)
                img_list.append(img)
                seg_list.append(seg)
                full_seg_list.append(full_seg)
                is_last_list.append(is_last[-1])

            image_batch = tf.stack(img_list, axis=0)
            seg_batch = tf.stack(seg_list, axis=0)
            full_seg_batch = tf.stack(full_seg_list, axis=0)
            is_last_batch = tf.stack(is_last_list, axis=0)
            dist_batch = tf.stack(dist_list, axis=0) if self.return_dist else None

            if self.data_format == 'NDHWC':
                image_batch = tf.expand_dims(image_batch, 5)
                seg_batch = tf.expand_dims(seg_batch, 5)
            elif self.data_format == 'NCDHW':
                image_batch = tf.expand_dims(image_batch, 2)
                seg_batch = tf.expand_dims(seg_batch, 2)
            else:
                raise ValueError()

            # image_batch_list = tf.unstack(image_batch, num=self.unroll_len, axis=0)
            # seg_batch_list = tf.unstack(seg_batch, num=self.unroll_len, axis=0)
            # full_seg_batch_list = tf.unstack(full_seg_batch, num=self.unroll_len, axis=0)

        if self.return_dist:
            # dist_batch_list = tf.unstack(dist_batch, num=self.unroll_len, axis=0)
            return image_batch, seg_batch, full_seg_batch, is_last_batch, dist_batch

        return image_batch, seg_batch, full_seg_batch, is_last_batch

    def start_queues(self, coord=tf.train.Coordinator(), debug=False):
        self._read_sequence_to_ram_()
        self.coord = coord
        threads = []
        for q, q_stat in zip(self.q_list, self.q_stat_list):
            if debug:
                self._load_and_enqueue(q, q_stat)
            for _ in range(self.num_threads):
                t = threading.Thread(target=self._load_and_enqueue, args=(q, q_stat))
                t.daemon = True
                t.start()
                threads.append(t)
                self.coord.register_thread(t)

        t = threading.Thread(target=self._monitor_queues_)
        t.daemon = True
        t.start()
        threads.append(t)
        self.coord.register_thread(t)
        return threads

    def _monitor_queues_(self):
        while not self.coord.should_stop():
            time.sleep(1)
        for q in self.q_list:
            q.close(cancel_pending_enqueues=True)

    def get_batch(self):
        return self._batch_queues_()


class CTCInferenceReaderTime(object):

    def __init__(self, data_path, filename_format='t*.tif', normalize=True, pre_sequence_frames=0):

        file_list = glob.glob(os.path.join(data_path, filename_format))
        if len(file_list) == 0:
            raise ValueError('Could not read images from: {}'.format(os.path.join(data_path, filename_format)))

        def gen():
            file_list.sort()
            # pre pad mirror image and post pat mirror image
            # file_list_pre = file_list[:pre_sequence_frames].copy()
            # file_list_pre.reverse()
            file_list_post = file_list[len(file_list)-pre_sequence_frames:].copy()
            file_list_post.reverse()

            # full_file_list = file_list_pre + file_list + file_list_post
            full_file_list = file_list + file_list_post
            full_file_list.reverse()
            for file in full_file_list:
                img = cv2.imread(file, -1).astype(np.float32)
                if img is None:
                    raise ValueError('Could not read image: {}'.format(file))
                if normalize:
                    img = (img - img.mean())
                    img = img / (img.std())
                yield img

        self.dataset = tf.data.Dataset.from_generator(gen, tf.float32)

    @classmethod
    def unit_test(cls):
        data_path = '/Users/aarbelle/Documents/CellTrackingChallenge/Training/DIC-C2DH-HeLa/01'
        filename_format = 't*.tif'
        normalize = True
        data_cls = cls(data_path, filename_format, normalize)
        for img, fname in data_cls.dataset:
            print(fname, img.shape, img.numpy().max(), img.numpy().min(), img.numpy().mean(), img.numpy().std())




class CTCInferenceReader(object):

    def __init__(self, data_path, filename_format='t*.tif', normalize=True, pre_sequence_frames=0):

        file_list = glob.glob(os.path.join(data_path, filename_format))
        if len(file_list) == 0:
            raise ValueError('Could not read images from: {}'.format(os.path.join(data_path, filename_format)))

        def gen():
            file_list.sort()

            file_list_pre = file_list[:pre_sequence_frames].copy()
            file_list_pre.reverse()
            full_file_list = file_list_pre + file_list
            for file in full_file_list:
                img = cv2.imread(file, -1).astype(np.float32)
                if img is None:
                    raise ValueError('Could not read image: {}'.format(file))
                if normalize:
                    img = (img - img.mean())
                    img = img / (img.std())
                yield img

        self.dataset = tf.data.Dataset.from_generator(gen, tf.float32)

    @classmethod
    def unit_test(cls):
        data_path = '/Users/aarbelle/Documents/CellTrackingChallenge/Training/DIC-C2DH-HeLa/01'
        filename_format = 't*.tif'
        normalize = True
        data_cls = cls(data_path, filename_format, normalize)
        for img, fname in data_cls.dataset:
            print(fname, img.shape, img.numpy().max(), img.numpy().min(), img.numpy().mean(), img.numpy().std())


class CTCInferenceReader3D(object):

    def __init__(self, data_path, filename_format='t*.tif', normalize=True, pre_sequence_frames=0,
                 max_size=(64, 128, 128)):

        file_list = glob.glob(os.path.join(data_path, filename_format))
        if len(file_list) == 0:
            raise ValueError('Could not read images from: {}'.format(os.path.join(data_path, filename_format)))
        crop = False

        def gen():
            file_list.sort()

            file_list_pre = file_list[:pre_sequence_frames].copy()
            file_list_pre.reverse()
            full_file_list = file_list_pre + file_list
            for file in full_file_list:
                img = utils.read_multi_tiff(file)
                if img is None:
                    raise ValueError('Could not read image: {}'.format(file))
                if normalize:
                    img = (img - img.mean())
                    img = img / (img.std())
                if crop:
                    crop_start = []
                    crop_end = []
                    for ind, (s, ms) in enumerate(zip(img.shape, max_size)):
                        if s > ms:
                            crop_a = int((s - ms) / 2)
                            crop_b = crop_a + ms
                            crop_start.append(crop_a)
                            crop_end.append(crop_b)
                        else:
                            crop_start.append(0)
                            crop_end.append(s)
                    img = img[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1],
                          crop_start[2] - max_size[2] - 60:crop_end[2] - max_size[2] - 60]
                yield img

        self.dataset = tf.data.Dataset.from_generator(gen, tf.float32)

    @classmethod
    def unit_test(cls):
        data_path = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Fluo-N3DH-SIM+/01'
        filename_format = 't*.tif'
        normalize = True
        data_cls = cls(data_path, filename_format, normalize)
        for img in data_cls.dataset:
            print(img.shape, img.numpy().max(), img.numpy().min(), img.numpy().mean(), img.numpy().std())


class CTCInferenceReader3DSlice(object):

    def __init__(self, data_path, filename_format='t*.tif', normalize=True, pre_sequence_frames=0, depth_pad=0):

        file_list = glob.glob(os.path.join(data_path, filename_format))
        if len(file_list) == 0:
            raise ValueError('Could not read images from: {}'.format(os.path.join(data_path, filename_format)))

        def gen():
            file_list.sort()

            file_list_pre = file_list[:pre_sequence_frames].copy()
            file_list_pre.reverse()
            full_file_list = file_list_pre + file_list
            for file in full_file_list:
                img = utils.read_multi_tiff(file)
                if img is None:
                    raise ValueError('Could not read image: {}'.format(file))
                if normalize:
                    img = (img - img.mean())
                    img = img / (img.std())
                if depth_pad:
                    img = np.pad(img, ((depth_pad, depth_pad), (0, 0), (0, 0)), 'reflect')
                yield img

        self.dataset = tf.data.Dataset.from_generator(gen, tf.float32)

    @classmethod
    def unit_test(cls):
        data_path = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Fluo-N3DH-SIM+/01'
        filename_format = 't*.tif'
        normalize = True
        data_cls = cls(data_path, filename_format, normalize)
        for img in data_cls.dataset:
            print(img.shape, img.numpy().max(), img.numpy().min(), img.numpy().mean(), img.numpy().std())


class CTCRAMReaderSequence3DSlice(object):
    def __init__(self, sequence_folder_list, image_crop_size=(128, 128), unroll_len=7, deal_with_end=0, batch_size=4,
                 queue_capacity=32, num_threads=3,
                 data_format='NCHW', randomize=True, return_dist=False, keep_sample=1, elastic_augmentation=True,
                 depth=0, load_to_ram=True, output_tra=False, erode_dilate_tra=(0,0)):
        if not isinstance(image_crop_size, tuple):
            image_crop_size = tuple(image_crop_size)
        self.coord = None
        self.unroll_len = unroll_len
        self.sequence_data = {}
        self.sequence_folder_list = sequence_folder_list
        self.elastic_augmentation = elastic_augmentation
        self.sub_seq_size = image_crop_size + (depth * 2 + 1,)
        self.dist_sub_seq_size = (2,) + image_crop_size
        self.deal_with_end = deal_with_end
        self.batch_size = batch_size
        self.queue_capacity = queue_capacity
        self.data_format = data_format
        self.randomize = randomize
        self.return_dist = return_dist
        self.num_threads = num_threads
        self.keep_sample = keep_sample
        self.switch_to_local_db = False
        self.depth = depth
        self.load_to_ram = load_to_ram
        self.ram_data = {}
        self.output_tra = output_tra
        self.erode_dilate_tra = erode_dilate_tra[0]

        self.q_list, self.q_stat_list = self._create_queues()
        np.random.seed(1)

    @classmethod
    def unit_test(cls):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/'
        dataset = 'Fluo-N3DL-DRO'
        sequence = '01'
        train_data = True
        image_crop_size = (256, 256)
        unroll_len = 3
        num_threads = 3
        batch_size = 4
        debug = True
        queue_capacity = 2000
        depth = 1
        erode_dilate_tra = (7,0)
        sequence_folder_list = [(os.path.join(root_dir, dataset), sequence, train_data)]
        data = cls(sequence_folder_list, batch_size=batch_size, image_crop_size=image_crop_size, unroll_len=unroll_len,
                   elastic_augmentation=True, num_threads=num_threads, queue_capacity=queue_capacity, depth=depth,
                   load_to_ram=False, output_tra=True, erode_dilate_tra=erode_dilate_tra)
        coord = tf.train.Coordinator()
        data.start_queues(coord, debug=debug)
        time.sleep(30)

        for i in range(1000):
            time.sleep(0.1)
            start = time.time()
            image_batch_list, seg_batch_list, full_seg_batch_list, is_last_batch, tra_batch_list = data.get_batch()
            print([qs().numpy() for qs in data.q_stat_list])
            print('Time: {}'.format(time.time() - start))
            # print([q.size().numpy() for q in data.q_list])
            print(image_batch_list[0].shape)
            print(tra_batch_list[0].shape)

    def _read_sequence_to_ram_(self):

        for sequence_folder in self.sequence_folder_list:
            train_set = True
            seq = '01'
            sequence_folder_orig = sequence_folder
            if isinstance(sequence_folder, tuple):
                sequence_folder, seq, train_set = sequence_folder

            utils.log_print('Reading Sequence {}'.format(sequence_folder))
            with open(os.path.join(sequence_folder, 'metadata_{}.pickle'.format(seq)), 'rb') as fobj:
                metadata = pickle.load(fobj)
            filename_list = metadata['filelist']
            original_size = 0
            keep_inds = []
            for t, f in enumerate(filename_list):
                if f[1] is not None and f[2] is True:
                    original_size += 1
                    keep_inds.append(t)
                elif isinstance(f[1], list):
                    for z, (slice_f, valid_f) in enumerate(zip(f[1], f[2])):
                        if slice_f is not None and valid_f is True:
                            original_size += 1
                            keep_inds.append(t)
                            break
            keep_inds = np.array(keep_inds)

            keep_rate = self.keep_sample if train_set else 0
            keep_num = np.round(keep_rate * original_size).astype(np.int16)
            keep_vec = np.zeros(len(filename_list))
            if keep_num > 0 and len(keep_inds) > 0:
                keep_inds = np.random.choice(keep_inds, keep_num, replace=False)
                keep_vec[keep_inds] = 1
            downampled_size = keep_num

            # for t, (filename, keep_seg) in enumerate(zip(filename_list, keep_vec)):
            #
            #     if not keep_seg:
            #         filename[1] = None
            #         filename[2] = None

            # if keep_rate < 1:
            #     print('Downsampling Training Segmentaiont with rate:{}. Original set size: {}. '
            #           'Downsampled set size: {}'.format(keep_rate, original_size, downampled_size))
            #
            self.sequence_data[sequence_folder_orig] = {'metadata': metadata}

    def _read_sequence_data(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        # if isinstance(sequence_folder, tuple):
        #     sequence_folder = sequence_folder[0]
        return self.sequence_data[sequence_folder], sequence_folder

    def _read_sequence_metadata(self):
        sequence_folder = random.choice(self.sequence_folder_list)
        if isinstance(sequence_folder, tuple):
            sequence_folder = sequence_folder[0]
        with open(os.path.join(sequence_folder, 'metadata.pickle'), 'rb') as fobj:
            metadata = pickle.load(fobj)
        return metadata

    @staticmethod
    def _get_elastic_affine_matrix_(shape_size, alpha_affine):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
         .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
              Convolutional Neural Networks applied to Visual Document Analysis", in
              Proc. of the International Conference on Document Analysis and
              Recognition, 2003.

          Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         """
        random_state = np.random.RandomState(None)

        # Random affine
        center_square = np.float32(shape_size[:2]) // 2
        square_size = min(shape_size[:2]) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        affine_matrix = cv2.getAffineTransform(pts1, pts2)

        return affine_matrix, random_state

    @staticmethod
    def _get_transformed_image_(image, affine_matrix, indices, seg=False):

        shape = image.shape
        if seg:
            trans_img = cv2.warpAffine(image, affine_matrix, shape[::-1], borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=-1, flags=cv2.INTER_NEAREST)
            trans_coord = map_coordinates(trans_img, indices, order=0, mode='constant', cval=-1).reshape(shape)
        else:
            trans_img = cv2.warpAffine(image, affine_matrix, shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
            trans_coord = map_coordinates(trans_img, indices, order=1, mode='reflect').reshape(shape)

        return trans_coord

    @staticmethod
    def _get_indices4elastic_transform(shape, alpha, sigma, random_state):
        dxr = random_state.rand(*shape)
        dx = gaussian_filter((dxr * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        return indices

    @staticmethod
    def _fix_transformed_segmentation(trans_seg):
        # separate close blobs with the edge of transformation
        # edge is pixels that get fractional value after transofrmation
        strel = np.zeros((3, 3))
        trans_seg = np.round(trans_seg)
        # region_props = skimage.measure.regionprops(trans_seg)
        # errosion = grey_erosion(trans_seg, np.zeros(3, 3, 3))
        dilation = grey_dilation(trans_seg.astype(np.int32), structure=strel.astype(np.int8))
        bw = np.minimum(trans_seg, 1)
        bw[np.logical_and(np.not_equal(trans_seg, dilation), np.greater(dilation, 0))] = 2

        return bw

    @staticmethod
    def _gt2dist_(gt_image):
        gt_fg = gt_image == 1
        _, labeled_gt = cv2.connectedComponents(gt_fg.astype(np.uint8))
        im_shape = gt_image.shape
        dist_1 = np.ones_like(gt_image) * (im_shape[0] + im_shape[1]) + 2.
        dist_2 = dist_1 + 1.

        for label in np.unique(labeled_gt):
            if label == 0:
                continue
            bw = np.equal(labeled_gt, label).astype(np.float32)
            bw_erode = cv2.erode(bw, np.ones((3, 3)))
            edge = np.logical_and(np.logical_not(bw_erode), bw)

            dist = distance_transform_edt(np.logical_not(edge))
            is_first_dist = np.less(dist, dist_1)
            dist_2[is_first_dist] = dist_1[is_first_dist]
            is_second_dist = np.logical_and(np.less(dist, dist_2), np.logical_not(is_first_dist))

            dist_1[is_first_dist] = dist[is_first_dist]
            dist_2[is_second_dist] = dist[is_second_dist]
        out = np.stack((dist_1, dist_2), 0)

        return out, (dist_1, dist_2)

    @staticmethod
    def _adjust_brightness_(image, delta):
        """
        Args:
        image (numpy.ndarray)
        delta
        """

        out_img = image + delta
        return out_img

    @staticmethod
    def _adjust_contrast_(image, factor):
        """
        Args:
        image (numpy.ndarray)
        factor
        """

        img_mean = image.mean()
        out_img = (image - img_mean) * factor + img_mean
        return out_img

    def _load_and_enqueue(self, q, q_stat, debug_timing=False):

        dist_crop = None
        unroll_len = self.unroll_len
        max_t_global = 1
        processed_image = np.zeros((max_t_global,) + self.sub_seq_size)
        processed_seg = np.zeros((max_t_global,) + self.sub_seq_size[:2] + (1,))
        processed_tra = np.zeros((max_t_global,) + self.sub_seq_size[:2] + (1,))
        full_seg_seq = np.zeros((max_t_global,))
        is_last = np.zeros((max_t_global,))
        try:
            while not self.coord.should_stop():
                all_times = []
                all_times.append(time.time())
                seq_data, sequence_folder = self._read_sequence_data()
                max_t = len(seq_data['metadata']['filelist'])
                max_depth = seq_data['metadata']['shape'][0]
                if max_t > max_t_global:
                    max_t_global = max_t
                    processed_image = np.zeros((max_t_global,) + self.sub_seq_size)
                    processed_seg = np.zeros((max_t_global,) + self.sub_seq_size[:2] + (1,))
                    processed_tra = np.zeros((max_t_global,) + self.sub_seq_size[:2] + (1,))
                    full_seg_seq = np.zeros((max_t_global,))
                    is_last = np.zeros((max_t_global,))
                # processed_image = np.zeros((max_t,) + self.sub_seq_size)
                # processed_seg = np.zeros((max_t,) + self.sub_seq_size[:2] + (1,))
                # full_seg_seq = np.zeros((max_t,))
                # is_last = np.zeros((max_t,))
                all_fnames = []

                all_times.append(time.time())
                img_size = seq_data['metadata']['shape'][1:]
                selected_z = np.random.randint(seq_data['metadata']['shape'][0])

                random_sub_sample = np.random.randint(1, 4) if self.randomize else 0
                random_reverse = np.random.randint(0, 2) if self.randomize else 0
                all_times.append(time.time())

                if img_size[0] - self.sub_seq_size[0] > 0:
                    crop_y = np.random.randint(0, img_size[0] - self.sub_seq_size[0]) if self.randomize else 0
                else:
                    crop_y = 0
                if img_size[1] - self.sub_seq_size[1] > 0:
                    crop_x = np.random.randint(0, img_size[1] - self.sub_seq_size[1]) if self.randomize else 0
                else:
                    crop_x = 0
                all_times.append(time.time())

                flip = np.random.randint(0, 2, 2) if self.randomize else [0, 0]
                rotate = np.random.randint(0, 4) if self.randomize else 0
                all_times.append(time.time())

                if self.elastic_augmentation:
                    affine_matrix, random_state = self._get_elastic_affine_matrix_(self.sub_seq_size,
                                                                                   self.sub_seq_size[1] * 0.08)
                    indices = self._get_indices4elastic_transform(self.sub_seq_size[:2], self.sub_seq_size[1] * 2,
                                                                  self.sub_seq_size[1] * 0.15,
                                                                  random_state)
                else:
                    affine_matrix = indices = None
                all_times.append(time.time())

                filename_idx = list(range(max_t))

                if random_reverse:
                    filename_idx.reverse()
                if random_sub_sample:
                    filename_idx = filename_idx[::random_sub_sample]
                seq_len = len(filename_idx)
                remainder = seq_len % unroll_len

                if remainder:
                    if self.deal_with_end == 0:
                        filename_idx = filename_idx[:-remainder]
                    elif self.deal_with_end == 1:
                        filename_idx += filename_idx[-2:-unroll_len + remainder - 2:-1]
                    elif self.deal_with_end == 2:
                        filename_idx += filename_idx[-1:] * (unroll_len - remainder)
                crop_y_stop = crop_y + self.sub_seq_size[0]
                crop_x_stop = crop_x + self.sub_seq_size[1]
                all_fnames = []
                start_z = np.maximum(selected_z - self.depth, 0)
                end_z = selected_z + self.depth + 1
                pad_start_z = pad_end_z = 0
                if end_z > max_depth:
                    pad_end_z = end_z - max_depth
                    end_z = max_depth
                if selected_z < self.depth:
                    pad_start_z = self.depth - selected_z

                if not self.load_to_ram:

                    crop = (crop_x, crop_y, crop_x_stop, crop_y_stop)
                    img_array = np.zeros((1+2*self.depth-pad_end_z-pad_start_z, self.sub_seq_size[0],
                                          self.sub_seq_size[1]))
                    # img_array = None
                else:
                    crop = None
                    img_array = None

                all_times.append(time.time())
                if debug_timing:
                    print('Start sequence timing: ', all_times[-1] - all_times[0])
                    all_times = np.array(all_times)
                    print('Detailed Start sequence timing: ', all_times[1:] - all_times[:-1])
                for t, file_idx in enumerate(filename_idx):

                    all_times = []
                    all_times.append(time.time())
                    if self.output_tra:
                        img, seg, full_seg, tra = self.read_images(sequence_folder, file_idx, start_z=start_z,
                                                                   stop_z=end_z, crop=crop, img=img_array)

                    else:
                        img, seg, full_seg = self.read_images(sequence_folder, file_idx, start_z=start_z,
                                                              stop_z=end_z, crop=crop)
                    all_times.append(time.time())

                    img_d, img_h, img_w = img.shape
                    # if img_d + pad_start_z < (1 + self.depth * 2):
                    #     pad_end_z = (1 + self.depth * 2) - (img_d + pad_start_z)

                    if pad_end_z or pad_start_z:
                        img = np.pad(img, ((pad_start_z, pad_end_z), (0, 0), (0, 0)), mode='reflect')
                        seg = np.pad(seg, ((pad_start_z, pad_end_z), (0, 0), (0, 0)), mode='constant',
                                     constant_values=-1)
                        if img.shape[1] < self.depth*2 + 1:
                            print('Problem in image shape {} from sequence: {}, crop: {}, '
                                                                   'selected_z: {}'.format(file_idx, sequence_folder,
                                                                                           crop, selected_z))
                        if self.output_tra:
                            tra = np.pad(tra, ((pad_start_z, pad_end_z), (0, 0), (0, 0)), mode='constant',
                                         constant_values=-1)
                    all_times.append(time.time())
                    if crop is not None:
                        img_crop_all = img
                        img_max = img.max()

                        seg_crop_all = seg
                        if self.output_tra:
                            tra_crop_all = tra
                        else:
                            tra_crop_all = np.zeros_like(seg_crop_all)
                    else:
                        img_crop_all = img[:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                        img_max = img.max()

                        seg_crop_all = seg[:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                        if self.output_tra:
                            tra_crop_all = tra[:, crop_y:crop_y_stop, crop_x:crop_x_stop]
                        else:
                            tra_crop_all = np.zeros_like(seg_crop_all)
                    all_times.append(time.time())

                    img_mod_all = []
                    seg_mod_all = []
                    tra_mod_all = []
                    random_constrast_factor = np.random.rand() + 0.5
                    random_brightness_delta = (np.random.rand() - 0.5) * 0.2 * img_max
                    all_times.append(time.time())
                    if debug_timing:
                        print('Start frames timing: ', all_times[-1] - all_times[0])
                        all_times = np.array(all_times)
                        print('Detailed Start frame timing: ', all_times[1:] - all_times[:-1])
                    for img_crop, seg_crop, tra_crop in zip(img_crop_all, seg_crop_all, tra_crop_all):
                        all_times = []
                        all_times.append(time.time())

                        # img_crop = np.transpose(img_crop, (1, 2, 0))
                        # seg_crop = np.transpose(seg_crop, (1, 2, 0))
                        filename = seq_data['metadata']['filelist'][file_idx][0]
                        if self.randomize:
                            # contrast factor between [0.5, 1.5]

                            # random brightness delta plus/minus 10% of maximum value
                            img_crop = self._adjust_contrast_(img_crop, random_constrast_factor)
                            img_crop = self._adjust_brightness_(img_crop, random_brightness_delta)
                        all_times.append(time.time())

                        if self.elastic_augmentation:
                            trans_img = self._get_transformed_image_(img_crop, affine_matrix, indices)
                            all_times.append(time.time())

                            img_crop = trans_img
                            if np.any(np.isnan(img_crop)):
                                self.coord.request_stop(ValueError('NaN in image {} from sequence: {}, crop: {}, '
                                                                   'selected_z: {}'.format(file_idx, sequence_folder,
                                                                                           crop, selected_z)))
                                return
                            if np.any(np.isinf(img_crop)):
                                self.coord.request_stop(ValueError('Inf in image {} from sequence: {}'
                                                                   ''.format(file_idx, sequence_folder)))
                                return
                            if not np.equal(seg_crop, -1).all():
                                seg_not_valid = np.equal(seg_crop, -1)
                                labeled_gt = seg_crop
                                labeled_gt[:, 0] = 0
                                labeled_gt[:, -1] = 0
                                labeled_gt[-1, :] = 0
                                labeled_gt[0, :] = 0
                                trans_seg = self._get_transformed_image_(labeled_gt.astype(np.float32), affine_matrix,
                                                                         indices, seg=True)
                                all_times.append(time.time())

                                trans_not_valid = self._get_transformed_image_(seg_not_valid.astype(np.float32),
                                                                               affine_matrix,
                                                                               indices, seg=True)
                                all_times.append(time.time())

                                trans_seg_fix = self._fix_transformed_segmentation(trans_seg)
                                trans_not_valid = np.logical_or(np.greater(trans_not_valid, 0.5),
                                                                np.equal(trans_seg, -1))
                                all_times.append(time.time())

                                seg_crop = trans_seg_fix
                                seg_crop[trans_not_valid] = -1
                                if np.any(np.isnan(seg_crop)):
                                    raise ValueError(
                                        'NaN in Seg {} from sequence: {}'.format(file_idx, sequence_folder))
                                if np.any(np.isinf(seg_crop)):
                                    self.coord.request_stop(ValueError(
                                        'Inf in Seg {} from sequence: {}'.format(file_idx, sequence_folder)))
                                    return
                            if self.output_tra:

                                tra_crop = self._get_transformed_image_(tra_crop.astype(np.float32), affine_matrix,
                                                                        indices, seg=True)
                                if np.any(np.isnan(tra_crop)):
                                    raise ValueError(
                                        'NaN in Tra {} from sequence: {}'.format(file_idx, sequence_folder))
                                if np.any(np.isinf(tra_crop)):
                                    raise ValueError(
                                        'Inf in Tra {} from sequence: {}'.format(file_idx, sequence_folder))
                        else:
                            seg_crop = self._fix_transformed_segmentation(seg_crop)

                        if flip[0]:
                            img_crop = cv2.flip(img_crop, 0)
                            seg_crop = cv2.flip(seg_crop, 0)
                        if flip[1]:
                            img_crop = cv2.flip(img_crop, 1)
                            seg_crop = cv2.flip(seg_crop, 1)
                        if rotate > 0:
                            img_crop = np.rot90(img_crop, rotate)
                            seg_crop = np.rot90(seg_crop, rotate)
                        if self.output_tra:
                            if flip[0]:
                                tra_crop = cv2.flip(tra_crop, 0)
                            if flip[1]:
                                tra_crop = cv2.flip(tra_crop, 1)
                            if rotate > 0:
                                tra_crop = np.rot90(tra_crop, rotate)

                        img_mod_all.append(img_crop)
                        seg_mod_all.append(seg_crop)
                        if self.output_tra:
                            if self.erode_dilate_tra < 0:
                                kernel = np.zeros((-self.erode_dilate_tra, -self.erode_dilate_tra),
                                                  dtype=tra_crop.dtype)
                                tra_crop = grey_erosion(tra_crop.astype(np.float32), structure=kernel)
                            elif self.erode_dilate_tra > 0:
                                kernel = np.zeros((self.erode_dilate_tra, self.erode_dilate_tra),
                                                  dtype=tra_crop.dtype)
                                tra_crop = grey_dilation(tra_crop.astype(np.float32), structure=kernel)
                            tra_mod_all.append(tra_crop)
                        all_times.append(time.time())

                        if debug_timing:
                            all_times = np.array(all_times)
                            print('Frame Times')
                            print(all_times[1:] - all_times[:-1])

                    img_crop = np.stack(img_mod_all, axis=2)
                    seg_crop = np.stack(seg_mod_all[self.depth:self.depth + 1], axis=2)
                    if self.output_tra:
                        tra_crop = np.stack(tra_mod_all[self.depth:self.depth + 1], axis=2)


                    is_last_frame = 1. if (t + 1) < len(filename_idx) else 0.
                    if self.num_threads == 1:
                        # print('full_seg', full_seg.shape)
                        all_times = []
                        all_times.append(time.time())
                        try:

                            while q_stat().numpy() > 0.9:
                                if self.coord.should_stop():
                                    return
                                time.sleep(1)

                            if self.output_tra:
                                q.enqueue(
                                    [img_crop, seg_crop, max(0, full_seg), is_last_frame, filename, tra_crop])
                            else:
                                q.enqueue([img_crop, seg_crop, max(0, full_seg), is_last_frame, filename])

                            all_times.append(time.time())

                            if debug_timing:
                                all_times = np.array(all_times)
                                print('Enqueue Times')
                                print(all_times[-1] - all_times[0])
                        except tf.errors.CancelledError:
                            pass
                    else:
                        is_last[t] = is_last_frame
                        processed_image[t] = img_crop
                        processed_seg[t] = seg_crop
                        if self.output_tra:
                            processed_tra[t] = tra_crop
                        all_fnames.append(filename)
                        full_seg_seq[t] = max(0, full_seg)
                        if self.coord.should_stop():
                            return
                if self.num_threads == 1:
                    continue

                try:
                    if self.num_threads > 1:
                        all_times = []
                        all_times.append(time.time())
                        # is_last = np.array(is_last)
                        # processed_image = np.array(processed_image)
                        # processed_seg = np.array(processed_seg)
                        # all_fnames = np.array(all_fnames)
                        # full_seg_seq = np.array(full_seg_seq)
                        all_times.append(time.time())
                        while q_stat().numpy() > 0.9:
                            if self.coord.should_stop():
                                return
                            time.sleep(1)
                        if self.output_tra:
                            q.enqueue_many([processed_image[:(t + 1)], processed_seg[:(t + 1)], full_seg_seq[:(t + 1)],
                                            is_last[:(t + 1)], all_fnames, processed_tra[:(t + 1)]])
                        else:
                            q.enqueue_many([processed_image[:(t + 1)], processed_seg[:(t + 1)], full_seg_seq[:(t + 1)],
                                            is_last[:(t + 1)], all_fnames])
                        all_times.append(time.time())

                        if debug_timing:
                            all_times = np.array(all_times)
                            print('Enqueue Times')
                            print(all_times[-1] - all_times[0], (all_times[1] - all_times[0]),
                                  (all_times[2] - all_times[1]), (all_times[2] - all_times[1]) / len(is_last))

                except tf.errors.CancelledError:
                    pass

        except tf.errors.CancelledError:
            pass

        except Exception as err:
            print('ERROR FROM DATA PROCESS')
            self.coord.request_stop(err)
            raise err

    def _create_queues(self):
        def normed_size(_q):
            @tf.function
            def q_stat():
                return tf.cast(_q.size(), tf.float32) / self.queue_capacity

            return q_stat

        with tf.name_scope('DataHandler'):
            dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.string]
            shapes = [self.sub_seq_size, self.sub_seq_size[:2] + (1,), (), (), ()]
            if self.output_tra:
                dtypes += [tf.float32]
                shapes += [self.sub_seq_size[:2] + (1,)]
            q_list = []

            q_stat_list = []
            for b in range(self.batch_size):
                q = tf.queue.FIFOQueue(self.queue_capacity, dtypes=dtypes, shapes=shapes, name='data_q_{}'.format(b))
                q_list.append(q)
                q_stat_list.append(normed_size(q))

        return q_list, q_stat_list

    def _batch_queues_(self):
        with tf.name_scope('DataHandler'):
            img_list = []
            seg_list = []
            tra_list = []
            full_seg_list = []
            is_last_list = []
            # fname_list = []
            for q in self.q_list:
                if self.output_tra:
                    img, seg, full_seg, is_last, fnames, tra = q.dequeue_many(self.unroll_len)
                    tra_list.append(tra)
                else:
                    img, seg, full_seg, is_last, fnames = q.dequeue_many(self.unroll_len)
                img_list.append(img)
                seg_list.append(seg)
                full_seg_list.append(full_seg)
                is_last_list.append(is_last[-1])
                # fname_list.append(fnames)

            image_batch = tf.stack(img_list, axis=0)
            seg_batch = tf.stack(seg_list, axis=0)
            # fnames_batch = tf.stack(fname_list, axis=0)
            full_seg_batch = tf.stack(full_seg_list, axis=0)
            is_last_batch = tf.stack(is_last_list, axis=0)
            tra_batch = tf.stack(tra_list, axis=0) if self.output_tra else None

            if self.data_format == 'NCHW':
                # image_batch = tf.expand_dims(image_batch, 4)
                # seg_batch = tf.expand_dims(seg_batch, 4)
                image_batch = tf.transpose(image_batch, (0, 1, 4, 2, 3))
                seg_batch = tf.transpose(seg_batch, (0, 1, 4, 2, 3))
                if self.output_tra:
                    tra_batch = tf.transpose(tra_batch, (0, 1, 4, 2, 3))
            elif not self.data_format == 'NHWC':
                raise ValueError('data_format should be either "NCHW" or "NHWC", but got: {}'.format(self.data_format))

        if self.output_tra:
            return image_batch, seg_batch, full_seg_batch, is_last_batch, tra_batch

        return image_batch, seg_batch, full_seg_batch, is_last_batch

    def _create_sequence_queue(self):
        sequence_queue = queue.Queue(maxsize=len(self.sequence_folder_list))
        for sequence in self.sequence_folder_list:
            sequence_queue.put(sequence)
        return sequence_queue

    def start_queues(self, coord=tf.train.Coordinator(), debug=False):
        self._read_sequence_to_ram_()
        threads = []
        self.coord = coord
        for q, q_stat in zip(self.q_list, self.q_stat_list):
            if debug:
                self._load_and_enqueue(q, q_stat, True)
            for _ in range(self.num_threads):
                t = threading.Thread(target=self._load_and_enqueue, args=(q, q_stat))
                t.daemon = True
                t.start()
                threads.append(t)
                self.coord.register_thread(t)

        t = threading.Thread(target=self._monitor_queues_)
        t.daemon = True
        t.start()
        self.coord.register_thread(t)
        threads.append(t)
        return threads

    def _monitor_queues_(self):
        while not self.coord.should_stop():
            time.sleep(1)
        for q in self.q_list:
            q.close(cancel_pending_enqueues=True)

    def get_batch(self):
        return self._batch_queues_()

    def read_images_old(self, sequence_folder, file_idx, start_z=None, stop_z=None):
        sequence_folder_orig = sequence_folder
        if isinstance(sequence_folder, tuple):
            sequence_folder, seq, train_set = sequence_folder
        if self.switch_to_local_db and os.path.exists(os.path.join(self.local_db_replace[1], '.done_data_move')):
            this_dir = sequence_folder.replace(self.local_db_replace[0], self.local_db_replace[1])
        else:
            this_dir = sequence_folder
        metadata = self.sequence_data[sequence_folder_orig]['metadata']
        filename_list = metadata['filelist']
        filename = filename_list[file_idx]
        img = utils.read_multi_tiff(os.path.join(this_dir, filename[0]), start_z=start_z, stop_z=stop_z)
        if img is None:
            print('Could Not Load Image: {}'.format(filename[0]))

        img = img.astype(np.float32)
        img = (img - img.mean()) / (img.std())
        seg = None
        full_seg = 1 if filename[2] is True else 0

        if filename[1] is None:
            seg = np.ones(img.shape[:3]) * (-1)
        elif not full_seg:
            if isinstance(filename[1], str):
                seg = utils.read_multi_tiff(os.path.join(this_dir, filename[1]), start_z=start_z, stop_z=stop_z)
                seg = seg.astype(np.float32)
            elif isinstance(filename[1], list):
                seg = np.ones(img.shape[:3]) * (-1)
                for slice_name, valid in zip(filename[1], filename[2]):
                    z = int(slice_name[-7:-4]) - start_z
                    if z < 0 or z >= seg.shape[0]:
                        continue
                    this_slice = cv2.imread(os.path.join(this_dir, slice_name), -1)
                    this_slice = this_slice.astype(np.float32)
                    if not valid or 'DRO' in sequence_folder:
                        this_slice[this_slice == 0] = -1
                    seg[z] = this_slice
                seg = seg

        else:
            seg = utils.read_multi_tiff(os.path.join(this_dir, filename[1]), start_z=start_z, stop_z=stop_z)

        return img, seg, full_seg

    def read_images(self, sequence_folder, file_idx, start_z=None, stop_z=None, crop=None, img=None):
        load_to_ram = self.load_to_ram
        sequence_folder_orig = sequence_folder
        if isinstance(sequence_folder, tuple):
            sequence_folder, seq, train_set = sequence_folder
        if self.switch_to_local_db and os.path.exists(os.path.join(self.local_db_replace[1], '.done_data_move')):
            this_dir = sequence_folder.replace(self.local_db_replace[0], self.local_db_replace[1])
        else:
            this_dir = sequence_folder
        metadata = self.sequence_data[sequence_folder_orig]['metadata']
        filename_list = metadata['filelist']
        filename = filename_list[file_idx]
        data_path = os.path.join(this_dir, filename[0])
        tra_data_path = tra = None
        if self.output_tra:
            tra_data_path = os.path.join(this_dir, filename[3]) if filename[3] else None
        if load_to_ram:
            if data_path not in self.ram_data.keys():
                full_img = utils.read_multi_tiff(data_path)
                if full_img is None:
                    print('Could Not Load Image: {}'.format(data_path))
                if data_path not in self.ram_data.keys():
                    self.ram_data[data_path] = full_img
                    print('Saved {} to RAM'.format(data_path))
            else:
                full_img = self.ram_data[data_path]
            start_z = max(0, start_z)
            stop_z = min(stop_z, full_img.shape[0])
            img = full_img[start_z:stop_z].copy()
            if self.output_tra:
                if tra_data_path is None:
                    tra = -1*np.ones_like(img)
                elif tra_data_path not in self.ram_data.keys():
                    full_tra = utils.read_multi_tiff(tra_data_path)
                    if full_tra is None:
                        print('Could Not Load Image: {}'.format(tra_data_path))
                    if tra_data_path not in self.ram_data.keys():
                        self.ram_data[tra_data_path] = full_tra
                        print('Saved {} to RAM'.format(tra_data_path))
                    tra = full_tra[start_z:stop_z].copy().astype(np.float32)

                else:
                    full_tra = self.ram_data[tra_data_path]
                    tra = full_tra[start_z:stop_z].copy().astype(np.float32)
                if (tra_data_path is not None) and not np.any(tra):
                    tra = -1 * np.ones_like(tra)
                if 'Fluo-N3DL-TRIC' in tra_data_path or 'Fluo-N3DL-DRO' in tra_data_path:
                    tra[tra == 0] = -1
        else:
            img = utils.read_multi_tiff(data_path, start_z=start_z, stop_z=stop_z, crop=crop, images_out=img)
            if self.output_tra:
                if tra_data_path is None:
                    tra = -1 * np.ones_like(img)
                else:
                    tra = utils.read_multi_tiff(tra_data_path, start_z=start_z, stop_z=stop_z, crop=crop).astype(np.float32)
                if tra_data_path is not None and not np.any(tra):
                    tra = -1 * np.ones_like(tra, dtype=np.float32)
        if img is None:
            print('Could Not Load Image: {}'.format(data_path))

        img = img.astype(np.float32)
        img = (img - img.mean()) / (img.std()+0.00001)
        seg = None
        full_seg = 1 if filename[2] is True else 0

        if filename[1] is None:
            seg = np.ones(img.shape[:3]) * (-1)
        elif not full_seg:
            if isinstance(filename[1], str):
                seg_path = os.path.join(this_dir, filename[1])
                if load_to_ram:
                    if seg_path not in self.ram_data.keys():
                        full_seg_img = utils.read_multi_tiff(seg_path).astype(np.float32)
                        if full_seg_img is None:
                            print('Could Not Load Image: {}'.format(seg_path))
                        self.ram_data[seg_path] = full_seg_img
                    else:
                        full_seg_img = self.ram_data[seg_path]
                    start_z = max(0, start_z)
                    stop_z = min(stop_z, full_seg_img.shape[0])
                    seg = full_seg_img[start_z:stop_z].copy()
                else:

                    seg = utils.read_multi_tiff(seg_path, start_z=start_z, stop_z=stop_z)
                    seg = seg.astype(np.float32)

            elif isinstance(filename[1], list):
                seg = np.ones(img.shape[:3]) * (-1)
                for slice_name, valid in zip(filename[1], filename[2]):
                    z = int(slice_name[-7:-4]) - start_z
                    if z < 0 or z >= seg.shape[0]:
                        continue
                    seg_path = os.path.join(this_dir, slice_name)

                    if load_to_ram:
                        if seg_path not in self.ram_data.keys():
                            this_slice = cv2.imread(seg_path, -1).astype(np.float32)
                            if this_slice is None:
                                print('Could Not Load Image: {}'.format(seg_path))
                            self.ram_data[seg_path] = this_slice
                        else:
                            this_slice = self.ram_data[seg_path]
                    else:
                        this_slice = cv2.imread(seg_path, -1).astype(np.float32)

                    if not valid or 'DRO' in sequence_folder:
                        this_slice[this_slice == 0] = -1
                    if crop is None:
                        seg[z] = this_slice
                    else:
                        (crop_x, crop_y, crop_x_stop, crop_y_stop) = crop
                        seg[z] = this_slice[crop_y:crop_y_stop, crop_x:crop_x_stop]
                seg = seg

        else:
            seg_path = os.path.join(this_dir, filename[1])
            if load_to_ram:
                if seg_path not in self.ram_data.keys():
                    full_seg_img = utils.read_multi_tiff(seg_path).astype(np.float32)
                    if full_seg_img is None:
                        print('Could Not Load Image: {}'.format(seg_path))
                    self.ram_data[seg_path] = full_seg_img
                else:
                    full_seg_img = self.ram_data[seg_path]
                start_z = max(0, start_z)
                stop_z = min(stop_z, full_seg_img.shape[0])
                seg = full_seg_img[start_z:stop_z].copy()
            else:

                seg = utils.read_multi_tiff(seg_path, start_z=start_z, stop_z=stop_z, crop=crop)
                seg = seg.astype(np.float32)
        if self.output_tra:
            return img, seg, full_seg, tra
        return img, seg, full_seg


if __name__ == "__main__":
    # CTCSegReaderSequence3D.unit_test()
    # LSCRAMReader2D.unit_test()
    # CTCInferenceReader3D.unit_test()
    # CTCRAMReaderSequence3DSlice.unit_test()
    CTCRAMReaderSequence2D.unit_test()
