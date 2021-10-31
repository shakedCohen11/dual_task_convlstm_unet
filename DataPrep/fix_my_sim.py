import os
import utils
import glob
import shutil
import imageio
import numpy as np
import cv2

path = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Fluo-N2DH-MYSIMV3/{:02d}_GT'
# path = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Fluo-N3DH-MySIM/{:02d}_GT'
seqs = 35
seg_filename = os.path.join('SEG', 'man_seg{:03d}.tif')
tra_filename = os.path.join('TRA', 'man_track{:03d}.tif')

for seq in range(45, 52):
    print(seq)
    t = 0
    while os.path.exists(os.path.join(path.format(seq),seg_filename.format(t))):
        # S = utils.read_multi_tiff(os.path.join(path.format(seq), seg_filename.format(t)))
        S = cv2.imread(os.path.join(path.format(seq), seg_filename.format(t)), -1)
        if S.dtype == np.float32:
            # imageio.mimwrite(os.path.join(path.format(seq), seg_filename.format(t)), S.astype(np.uint16))
            cv2.imwrite(os.path.join(path.format(seq), seg_filename.format(t)), S.astype(np.uint16))

            shutil.copy2(os.path.join(path.format(seq), seg_filename.format(t)), os.path.join(path.format(seq),
                                                                                              tra_filename.format(t)))

        t +=1
