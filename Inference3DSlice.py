import argparse
import os
import pickle
import scipy.ndimage
import numpy as np
# noinspection PyPackageRequirements
import tensorflow as tf
import Networks as Nets
import Params
from distutils.util import strtobool
import DataHandeling
import sys
from utils import log_print, get_model, bbox_crop, bbox_fill, get_random_color_pallete, draw_labeled_on_img
import imageio
import skfmm

# import seaborn
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

__author__ = 'arbellea@post.bgu.ac.il'
if not tf.__version__.split('.')[0] == '2':
    raise ImportError(f'Required tensorflow version 2.x. current version is: {tf.__version__}')

k = tf.keras


# noinspection PyArgumentList
def inference():
    # Load Model
    with open(os.path.join(params.model_path, 'model_params.pickle'), 'rb') as fobj:
        model_dict = pickle.load(fobj)
        input_depth = model_dict['input_depth']
    model_cls = get_model(model_dict['name'])

    device = '/gpu:0' if params.gpu_id >= 0 else '/cpu:0'
    with tf.device(device):
        add_binary_output = model_dict.get('add_binary_output', False) and not params.one_object
        model = model_cls(*model_dict['params'], data_format=params.data_format, pad_image=True,
                          add_binary_output=add_binary_output)
        model.load_weights(os.path.join(params.model_path, 'model.ckpt'))
        log_print("Restored from {}".format(os.path.join(params.model_path, 'model.ckpt')))
    base_out_temp_vis_fname = base_out_temp_label_fname = base_out_fname = base_out_temp_label_color_fname = None
    base_out_temp_contour_color_fname = base_out_temp_centers_hard_fname = base_out_temp_centers_fname = None
    if not params.dry_run:
        if params.save_intermediate and params.save_intermediate_path:
            base_out_temp_vis_fname = os.path.join(params.save_intermediate_vis_path, 'softmax{time:03d}.tif')
            base_out_temp_label_fname = os.path.join(params.save_intermediate_label_path, 'mask{time:03d}.tif')
            base_out_temp_label_color_fname = os.path.join(params.save_intermediate_label_path,
                                                           'mask_color{time:03d}.tif')
            base_out_temp_contour_color_fname = os.path.join(params.save_intermediate_contour_path,
                                                             'contour{time:03d}.tif')
            base_out_temp_centers_hard_fname = os.path.join(params.save_intermediate_centers_hard_path,
                                                            'centers_hard{time:03d}.tif')
            base_out_temp_centers_fname = os.path.join(params.save_intermediate_centers_path,
                                                       'centers{time:03d}.tif')
        base_out_fname = os.path.join(params.output_path, 'mask{time:03d}.tif')
    dataset = params.data_reader(params.sequence_path, params.filename_format,
                                 pre_sequence_frames=params.pre_sequence_frames,
                                 depth_pad=input_depth).dataset
    colors = get_random_color_pallete(0)

    try:
        for T, image in enumerate(dataset):
            t = T - params.pre_sequence_frames
            image_orig = image.numpy()[input_depth:-input_depth]
            image_shape = image.shape
            if len(image_shape) == 3:
                if params.data_format == 'NCHW':  # DNTHW
                    image = tf.reshape(image, [image_shape[0], 1, 1, image_shape[1], image_shape[2]])
                else:  # DNTHW
                    image = tf.reshape(image, [image_shape[0], 1, 1, image_shape[1], image_shape[2]])
            elif len(image_shape) == 4:
                image = tf.reshape(image, [1, 1, image_shape[1], image_shape[2], image_shape[3]])
            else:
                raise ValueError()
            image_softmax = []
            center_sigmoid = []
            for ind in range(input_depth, image.shape[0] - input_depth):
                start_depth = ind - input_depth
                end_depth = ind + input_depth + 1
                img_slice = image[start_depth:end_depth]
                if params.data_format == 'NCHW':  # DNTCHW
                    img_slice = tf.transpose(img_slice, (1, 2, 0, 3, 4))
                else:
                    img_slice = tf.transpose(img_slice, (1, 2, 3, 4, 0))

                if add_binary_output:
                    _, image_slice_softmax, center_slice_sigmoid, _ = model(img_slice, training=False)
                    center_sigmoid_np = center_slice_sigmoid.numpy().squeeze()
                    center_sigmoid.append(center_sigmoid_np)
                else:
                    _, image_slice_softmax = model(img_slice, training=False)

                image_softmax.append(image_slice_softmax.numpy())
            image_softmax = np.stack(image_softmax, axis=0)
            image_softmax_np = np.squeeze(image_softmax, (1, 2))
            if add_binary_output:
                center_sigmoid = np.stack(center_sigmoid, axis=0)
                centers_hard = np.greater(center_sigmoid, params.centers_sigmoid_threshold)
                centers_labels, num_centers = scipy.ndimage.label(centers_hard.astype(np.uint8))
            else:
                centers_labels = centers_hard = center_sigmoid = None
            if t < 0:
                continue

            if not params.dry_run:
                if params.data_format == 'NCHW':
                    seg_edge = np.greater_equal(image_softmax_np[:, 2], params.edge_thresh)
                    seg_cell_soft = image_softmax_np[:, 1]
                else:
                    seg_cell_soft = image_softmax_np[..., 1]
                    seg_edge = np.greater_equal(image_softmax_np[..., 2], params.edge_thresh)
                seg_cell = np.logical_and(np.equal(np.argmax(image_softmax_np, params.channel_axis),
                                                   1).astype(np.float32),
                                          np.logical_not(seg_edge))
                seg_edge = seg_edge.astype(np.float32)
                seg_cell = scipy.ndimage.morphology.binary_fill_holes(seg_cell).astype(np.float32)
                seg_edge = np.maximum((seg_edge - seg_cell), 0)
                if params.one_object:
                    labels = seg_cell.astype(np.uint8)
                    num_cells = 1
                else:
                    labels, num_cells = scipy.ndimage.label(seg_cell.astype(np.uint8))
                # cc_out = cv2.connectedComponentsWithStats(seg_cell.astype(np.uint8), 8, cv2.CV_32S)
                # num_cells = cc_out[0]
                # labels = cc_out[1]
                # stats = cc_out[2]
                # num_cells = labels.max()

                # dist, ind = scipy.ndimage.morphology.distance_transform_edt(1 - seg_cell, return_indices=True)
                # labels = labels[ind[0], ind[1], ind[2]] * seg_edge * (dist < params.edge_dist) + labels
                global_num_cells = num_cells
                # noinspection PyArgumentList
                areas, bins = np.histogram(labels.ravel(), np.arange(1, labels.max() + 2))
                b2 = bins[:-1]
                b3 = b2[areas < params.min_cell_size]
                labels[np.isin(labels, b3)] = 0
                for n_id, n in enumerate(np.unique(labels)):

                    if n == 0:
                        continue

                    bw = labels == n
                    if not np.any(bw):
                        continue

                    bw_crop, loc = bbox_crop(bw, three_d=True, margin=np.maximum(5, params.edge_dist))
                    dist, ind = scipy.ndimage.morphology.distance_transform_edt(1 - bw_crop, return_indices=True)
                    seg_edge_crop = seg_edge[loc[0]:loc[1], loc[2]:loc[3], loc[4]:loc[5]]
                    bw_crop2 = np.logical_or(bw_crop, np.logical_and(np.less_equal(dist, params.edge_dist),
                                                                     seg_edge_crop))

                    fill_crop = scipy.ndimage.morphology.binary_fill_holes(bw_crop2).astype(np.float32)
                    if fill_crop.sum() < params.min_cell_size:
                        continue
                    fill_diff = fill_crop - bw_crop
                    bw_fill = bbox_fill(bw, fill_diff, loc, three_d=True)
                    labels = labels + bw_fill * n

                    if add_binary_output and not params.no_tra:
                        centers_crop = centers_labels[loc[0]:loc[1], loc[2]:loc[3], loc[4]:loc[5]]
                        unique_centers = np.unique(centers_crop[(centers_crop * fill_crop) > 0])
                        unique_centers = unique_centers[unique_centers > 0]
                        if len(unique_centers) > 1:
                            seg_cell_soft_crop = seg_cell_soft[loc[0]:loc[1], loc[2]:loc[3], loc[4]:loc[5]]
                            seg_cell_soft_crop = seg_cell_soft_crop * fill_crop

                            order = 2

                            d_c = []
                            d_1 = []
                            while True:
                                try:
                                    d_c = []
                                    d_0 = []
                                    d_1 = []
                                    for c in unique_centers:
                                        if np.sum(centers_crop == c) < params.min_center_size:
                                            continue
                                        this_c = (centers_crop == c) * fill_crop
                                        phi = np.ma.MaskedArray(1 - 2 * this_c, np.logical_not(fill_crop))
                                        if order > 0:
                                            d1 = skfmm.travel_time(phi, seg_cell_soft_crop, order=order) * (
                                                    1 - this_c)
                                            d0 = skfmm.travel_time(phi, np.ones_like(phi), order=order) * (1 - this_c)
                                        else:
                                            d1 = skfmm.distance(phi) * (
                                                    1 - this_c)
                                            d0 = 0
                                        d_c.append(d1 - d0)
                                        d_0.append(d0)
                                        d_1.append(d1)
                                    break
                                except (RuntimeError, ValueError):

                                    order -= 1
                                    if order < 0:
                                        break

                            if len(d_c) > 1:
                                d_c = np.stack(d_c, 0)
                                new_label = np.argmin(d_c, axis=0) + 1
                                new_label = new_label * fill_crop
                                skip = False
                                for nc in range(len(d_c)):
                                    _, num_new_l_c = scipy.ndimage.label((new_label == nc).astype(np.uint8))
                                    if num_new_l_c > 2:
                                        skip = True
                                        break
                                if skip:
                                    new_label = np.argmin(d_1, axis=0) + 1
                                    new_label = new_label * fill_crop

                                new_label = bbox_fill(np.zeros_like(bw, dtype=np.uint16), new_label, loc, three_d=True)
                                for nl in np.unique(new_label):
                                    if nl <= 1:
                                        continue
                                    global_num_cells += 1
                                    new_global_l = global_num_cells
                                    labels[new_label == nl] = new_global_l

                # filter by fov
                if params.FOV:
                    fov_im = np.ones_like(labels)
                    fov_im[:, params.FOV, :] = 0
                    fov_im[:, -params.FOV:, :] = 0
                    fov_im[:, :, params.FOV] = 0
                    fov_im[:, :, -params.FOV:] = 0
                    fov_labels = labels * fov_im
                    unique_fov_labels = np.unique(fov_labels.flatten())
                    remove_ind = np.setdiff1d(np.arange(num_cells), unique_fov_labels)
                else:
                    remove_ind = []
                if params.save_intermediate:
                    #
                    if params.data_format == 'NCHW':
                        image_softmax_np = np.transpose(image_softmax_np, (0, 2, 3, 1))
                    out_fname = base_out_temp_vis_fname.format(time=t)
                    sigoutnp_vis = np.round(image_softmax_np * (2 ** 8 - 1)).astype(np.uint16)
                    imageio.mimwrite(out_fname, sigoutnp_vis.astype(np.uint8))
                    log_print("Saved File: {}".format(out_fname))

                labels_out = np.zeros_like(labels, dtype=np.uint16)
                labels_out_color = np.zeros(labels.shape + (4,), dtype=np.float32)

                # noinspection PyArgumentList
                areas, bins = np.histogram(labels.ravel(), np.arange(1, labels.max() + 2))
                b2 = bins[:-1]
                b3 = b2[np.logical_or(areas < params.min_cell_size, areas > params.max_cell_size)]
                labels[np.isin(labels, b3)] = 0

                p = 0

                for n in np.unique(labels):
                    if n == 0 or (n in remove_ind):
                        continue
                    p += 1
                    labels_out[labels == n] = p
                    labels_out_color[labels == n] = np.concatenate([colors[np.mod(p, len(colors))], np.array([1])])

                out_fname = base_out_fname.format(time=t)
                imageio.mimwrite(out_fname, labels_out.astype(np.uint16))
                log_print("Saved File: {}".format(out_fname))
                if params.save_intermediate:
                    merge = []
                    im_min = image_orig.min()
                    im_max = image_orig.max()
                    for im_or, lab in zip(image_orig, labels_out):
                        merge.append(draw_labeled_on_img(im_or, lab, im_min=im_min, im_max=im_max))
                    merge = np.stack(merge, axis=0)
                    out_fname = base_out_temp_label_fname.format(time=t)
                    imageio.mimwrite(out_fname, labels_out.astype(np.uint16))
                    log_print("Saved File: {}".format(out_fname))
                    out_fname = base_out_temp_contour_color_fname.format(time=t)
                    imageio.mimwrite(out_fname, merge.astype(np.uint8))
                    log_print("Saved File: {}".format(out_fname))
                    out_fname = base_out_temp_label_color_fname.format(time=t)
                    imageio.mimwrite(out_fname, (labels_out_color * 255).astype(np.uint8))
                    log_print("Saved File: {}".format(out_fname))
                    if add_binary_output:
                        out_fname = base_out_temp_centers_hard_fname.format(time=t)
                        imageio.mimwrite(out_fname, (centers_hard * 255).astype(np.uint8))
                        log_print("Saved File: {}".format(out_fname))
                        out_fname = base_out_temp_centers_fname.format(time=t)
                        imageio.mimwrite(out_fname, (center_sigmoid * 255).astype(np.uint8))
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
                            help="Visible GPUs: example, '0,2,3', use -1 for CPU")
    arg_parser.add_argument('--model_path', dest='model_path', type=str,
                            help="Path to trained model generated by train2D.py, folder should contain model.ckpt.*")
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
    arg_parser.add_argument('--max_cell_size', dest='max_cell_size', type=int,
                            help="Maximum cell size")
    arg_parser.add_argument('--fov', dest='FOV', type=int,
                            help="Feild of veiw for detection")
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
    arg_parser.add_argument('--one_object', dest='one_object', action='store_const', const=True,
                            help="Use if known that there is only one cell")
    arg_parser.add_argument('--dont_save_intermediate', dest='save_intermediate', action='store_const', const=False,
                            help="Do not save intermediate files")
    arg_parser.add_argument('--save_intermediate_path', dest='save_intermediate_path', type=str,
                            help="Path to save intermediate files, used only with --save_intermediate")
    arg_parser.add_argument('--dry_run', dest='dry_run', action='store_const', const=True,
                            help="Do not write any outputs: for debugging only")
    sys_args = sys.argv

    input_args = arg_parser.parse_args()
    args_dict = {key: val for key, val in vars(input_args).items() if not (val is None)}
    params = Params.CTCInferenceParams3DSlice(args_dict)
    tf_eps = tf.constant(1E-8, name='epsilon')
    try:
        inference()
    finally:
        log_print('Done')
