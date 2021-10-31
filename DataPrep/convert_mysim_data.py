import os
import utils
import glob
import shutil
import imageio
import numpy as np

print = utils.log_print
PATH = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/MySIM/'
OUT_PATH = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training'
SEG_FORMAT = './{:02d}_GT/SEG/man_seg{:03d}.tif'
TRA_FORMAT = './{:02d}_GT/TRA/man_track{:03d}.tif'
IM_FORMAT = './{:02d}/t{:03d}.tif'
skip = ['/media/rrtammyfs/labDatabase/CellTrackingChallenge/MySIM/51',
        '/media/rrtammyfs/labDatabase/CellTrackingChallenge/MySIM/43',
        '/media/rrtammyfs/labDatabase/CellTrackingChallenge/MySIM/24',
        '/media/rrtammyfs/labDatabase/CellTrackingChallenge/MySIM/18',
        '/media/rrtammyfs/labDatabase/CellTrackingChallenge/MySIM/19',
        '/media/rrtammyfs/labDatabase/CellTrackingChallenge/MySIM/38']


def main(path=PATH, out_path=OUT_PATH, im_format=IM_FORMAT, seg_format=SEG_FORMAT, tra_format=TRA_FORMAT,
         dataset_name='Fluo-N3DH-MySIM'):
    out_dataset_dir = os.path.join(out_path, dataset_name)
    os.makedirs(out_dataset_dir, exist_ok=True)
    dir_list = glob.glob(os.path.join(path, '*'))
    dir_list = [d for d in dir_list if os.path.isdir(d)]
    seq = len(skip) + 1
    for d in dir_list:
        if d in skip:
            continue
        print('Reading Sequence from:', d)
        images_path = os.path.join(d, 'image_all_tif', 'final', 'final{:03d}.tif')
        seg_path = os.path.join(d, 'image_all_tif', 'phantom', 'sceneMasks{:03d}.tif')
        t = 0
        im = utils.read_multi_tiff(images_path.format(t))
        seg = utils.read_multi_tiff(seg_path.format(t))
        if seg.shape[0] % im.shape[0]:
            print('Skipping Sequence from:', d)
            continue
        else:
            seg_down_rate = int(seg.shape[0] / im.shape[0])
        im_out_path = os.path.join(out_dataset_dir, im_format)
        seg_out_path = os.path.join(out_dataset_dir, seg_format)
        tra_out_path = os.path.join(out_dataset_dir, tra_format)
        os.makedirs(os.path.dirname(im_out_path.format(seq, 0)), exist_ok=True)
        os.makedirs(os.path.dirname(seg_out_path.format(seq, 0)), exist_ok=True)
        os.makedirs(os.path.dirname(tra_out_path.format(seq, 0)), exist_ok=True)

        t = 0
        while os.path.exists(images_path.format(t)):
            shutil.copy2(images_path.format(t), im_out_path.format(seq, t))
            seg = utils.read_multi_tiff(seg_path.format(t))
            seg_down = seg[::seg_down_rate]
            imageio.mimwrite(seg_out_path.format(seq, t), seg_down.astype(np.uint16))
            shutil.copy2(seg_out_path.format(seq, t), tra_out_path.format(seq, t))
            t += 1
        seq += 1
    print('Done')


if __name__ == '__main__':
    main()
