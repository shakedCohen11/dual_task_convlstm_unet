import sys

import numpy as np
import matplotlib.pyplot as plt
import utils
import cv2
import os
import pickle
from scipy.ndimage import grey_dilation
import argparse


def main(root_dir, seq, raw_file_template, seg_file_template, seg_3d, tra_file_template=None, force=False,
         force_yes_to_all=False, FOV=80):
    if os.path.exists(os.path.join(root_dir, 'metadata_{}.pickle'.format(seq))) and not force:
        raise FileExistsError('File {} already exists! use --force option to overwrite')

    all_image_files = os.listdir(os.path.join(root_dir, os.path.join(os.path.dirname(raw_file_template))))
    all_seg_files = os.listdir(os.path.join(root_dir, os.path.join(os.path.dirname(seg_file_template))))
    all_tra_files = os.listdir(os.path.join(root_dir, os.path.join(os.path.dirname(tra_file_template))))
    t = 0
    seq_metadata = {'filelist': [], 'max': 0, 'min': np.inf, 'shape': None}
    valid_seg = None if not force_yes_to_all else 'yes to all'
    while True:
        if os.path.basename(raw_file_template.format(t)) in all_image_files:

            im_fname = raw_file_template.format(t)
            tra_fname = tra_file_template.format(t) if tra_file_template is not None else None
            tra_fname = tra_fname if os.path.basename(tra_fname) in all_tra_files else None
            im = utils.read_multi_tiff(os.path.join(root_dir, im_fname))
            seq_metadata['max'] = np.maximum(seq_metadata['max'], im.max())
            seq_metadata['min'] = np.minimum(seq_metadata['min'], im.min())
            if seq_metadata['shape'] is None:
                seq_metadata['shape'] = im.shape
            elif not np.all(seq_metadata['shape'] == im.shape):
                raise ValueError(
                    'Image shape should be consistent for full sequence, expected {}, got {} for image {}'.format(
                        seq_metadata['shape'], im.shape, im_fname))
            if seg_3d:
                seg_fname = seg_file_template.format(t)
                if os.path.basename(seg_file_template.format(t)) in all_seg_files:
                    seg = utils.read_multi_tiff(os.path.join(root_dir, seg_fname))
                    all_valid = True
                    if valid_seg == 'yes to all':
                        pass
                    else:

                        for d, (im_s, seg_s) in enumerate(zip(im, seg)):
                            if valid_seg == 'yes to all':
                                break
                            im_s = (im_s - im_s.min()) / (im_s.max() - im_s.min())
                            imR = im_s.copy()
                            imG = im_s.copy()
                            imB = im_s.copy()
                            strel = np.zeros((5, 5))
                            dilation = grey_dilation(seg_s.astype(np.int32), structure=strel.astype(np.int8))
                            seg_boundary = np.zeros_like(im_s, dtype=np.bool)
                            seg_boundary[np.logical_and(np.not_equal(seg_s, dilation), np.greater(dilation, 0))] = True

                            imR[seg_boundary] = 1
                            imG[seg_boundary] = 0
                            imB[seg_boundary] = 0
                            imrgb = np.stack([imR, imG, imB], 2)
                            plt.figure(1)
                            plt.cla()
                            plt.imshow(imrgb)
                            plt.title('T = {}, Z = {}'.format(t, d))
                            plt.pause(0.1)
                            valid_seg = input(
                                'Frame {} Depth {}: Are all cells in the frame annotated [Y/n/yes to all]?'
                                ''.format(t, d)).lower()
                            all_valid = valid_seg in ['', 'y', 'yes', 'yes to all'] and all_valid

                    row = (im_fname, seg_fname, (valid_seg in ['', 'y', 'yes', 'yes to all']) or all_valid, tra_fname)
                else:
                    row = (im_fname, None, None, tra_fname)
            else:

                row = (im_fname, None, None, tra_fname)
                seg_fname_list = []
                valid_seg_list = []
                for d, im_d in enumerate(im):
                    seg_fname = seg_file_template.format(t, d)
                    if os.path.basename(seg_file_template.format(t, d)) in all_seg_files:
                        seg = cv2.imread(os.path.join(root_dir, seg_fname), -1)
                        seg_fname_list.append(seg_fname)
                        if valid_seg in ['yes to all', 'no to all']:
                            pass
                        else:
                            im_d = (im_d - im_d.min()) / (im_d.max() - im_d.min())
                            imR = im_d.copy()
                            imG = im_d.copy()
                            imB = im_d.copy()
                            strel = np.zeros((5, 5))
                            dilation = grey_dilation(seg.astype(np.int32), structure=strel.astype(np.int8))
                            seg_boundary = np.zeros_like(im_d, dtype=np.bool)
                            seg_boundary[np.logical_and(np.not_equal(seg, dilation), np.greater(dilation, 0))] = True

                            imR[seg_boundary] = 1
                            imG[seg_boundary] = 0
                            imB[seg_boundary] = 0
                            imrgb = np.stack([imR, imG, imB], 2)
                            plt.figure(1)
                            plt.cla()
                            plt.imshow(imrgb)
                            plt.title('T = {}, Z = {}'.format(t, d))
                            plt.pause(0.1)
                            valid_seg = input(
                                'Frame {} z {}: Are all cells in the frame annotated [Y/n/yes to all/no to all]? '
                                ''.format(t, d)).lower()
                            valid_seg_list.append(valid_seg in ['', 'y', 'yes', 'yes to all'])

                        row = (im_fname, seg_fname_list, valid_seg_list, tra_fname)

            seq_metadata['filelist'].append(row)
            print(row)

        else:
            break
        t += 1
    print(seq_metadata['filelist'])
    with open(os.path.join(root_dir, 'metadata_{}.pickle'.format(seq)), 'wb') as f:
        pickle.dump(seq_metadata, f, pickle.HIGHEST_PROTOCOL)


def get_default_run():
    root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training'
    dataset = 'Fluo-N3DL-TRIC'
    seq = '01'
    seg_3d = False
    force_yes_to_all = False

    raw_file_template = os.path.join(seq, 't{:03d}.tif')
    seg_file_template = os.path.join('{}_GT'.format(seq), 'SEG', 'man_seg{:03d}.tif')
    seg_file_template = os.path.join('{}_GT'.format(seq), 'SEG', 'man_seg_{:03d}.tif')
    tra_file_template = os.path.join('{}_GT'.format(seq), 'TRA',
                                     'man_track{:03d}.tif')  # Optional
    # dataset = 'Fluo-C3DH-H157'
    # seq = '01'
    # seg_3d = False
    # force_yes_to_all = False
    # raw_file_template = os.path.join(seq, 't{:03d}.tif')
    # seg_file_template = os.path.join('{}_GT'.format(seq), 'SEG', 'man_seg_{:03d}_{:03d}.tif')
    # tra_file_template = os.path.join('{}_GT'.format(seq), 'TRA',
    #                                  'man_track{:03d}.tif')  # Optional

    # seq_metadata[filelist] will be a list of tuples holding (raw_fname, seg_fname, tra_filename, is_seg_val)
    root_dir = os.path.join(root_dir, dataset)

    return root_dir, seq, raw_file_template, seg_file_template, tra_file_template, seg_3d, force_yes_to_all


if __name__ == '__main__':

    root_dir_, seq_, raw_file_template_, seg_file_template_, tra_file_template_, seg_3d_, force_yes_to_all_ = get_default_run()

    arg_parser = argparse.ArgumentParser(description='Run Train LSTMUnet Segmentation')
    arg_parser.add_argument('--root_dir', dest='root_dir', type=str,
                            help="Root directory of sequence, example: '~/CellTrackingChallenge/Train/Fluo-N2DH-SIM+")
    arg_parser.add_argument('--seq', dest='seq', type=str,
                            help="Sequence number (two digit) , example: '01' or '02' ")
    arg_parser.add_argument('--raw_file_template', dest='raw_file_template', type=str,
                            help="Template for image sequences, example: '01/t{:03d}.tif' where {:03d} will be replaced by a three digit number 000,001,...")
    arg_parser.add_argument('--seg_file_template', dest='seg_file_template', type=str,
                            help="Template for image sequences segmentation , example: '01_GT/SEG/man_seg{:03d}.tif' where {:03d} will be replaced by a three digit number 000,001,...")
    arg_parser.add_argument('--seg_3d', dest='seg_3d', type=str,
                            help="Template for image sequences segmentation , example: '01_GT/SEG/man_seg{:03d}.tif' where {:03d} will be replaced by a three digit number 000,001,...")
    arg_parser.add_argument('--tra_file_template', dest='tra_file_template', type=str,
                            help="Optional!. Template for image sequences tracking lables , example: '01_GT/TRA/man_track{:03d}.tif' where {:03d} will be replaced by a three digit number 000,001,...")

    arg_parser.add_argument('--force', dest='force', help='Force overwrite existing metadata pickle')
    sys_args = sys.argv

    input_args = arg_parser.parse_args()
    if len(sys_args) > 1:

        if sys_args < 5:
            raise SyntaxError('Please input all parameters: root_dir, raw_file_template, seg_file_'
                              'template and optionaly tra_file_template ')
        root_dir_ = input_args.root_dir
        raw_file_template_ = input_args.raw_file_template
        seg_file_template_ = input_args.seg_file_template
        tra_file_template_ = input_args.tra_file_template
    input_args.force = True
    force_yes_to_all_ = True
    seg_3d_ = True

    for s in range(1, 3):
        root_dir = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Silver_GT'
        # dataset = 'Fluo-N3DH-CHO'
        # dataset = 'Fluo-C3DL-MDA231'
        dataset = 'Fluo-C3DH-A549'
        root_dir_ = os.path.join(root_dir, dataset)

        seq_ = '{:02d}'.format(s)
        raw_file_template_ = os.path.join(seq_, 't{:03d}.tif')
        seg_file_template_ = os.path.join('{}_ST'.format(seq_), 'SEG', 'man_seg{:03d}.tif')
        # seg_file_template_ = os.path.join('{}_GT'.format(seq_), 'SEG', 'man_seg_{:03d}_{:03d}.tif')
        tra_file_template_ = os.path.join('{}_GT'.format(seq_), 'TRA', 'man_track{:03d}.tif')
        main(root_dir_, seq_, raw_file_template_, seg_file_template_, seg_3d_, tra_file_template_, input_args.force,
             force_yes_to_all=force_yes_to_all_)
