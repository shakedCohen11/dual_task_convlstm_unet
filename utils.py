from PIL import ImageSequence, Image
import numpy as np
from datetime import datetime
import Networks as Nets
import Networks3D as Nets3D
import traceback
import sys
from scipy.ndimage.morphology import grey_erosion, grey_dilation
try:
    import seaborn
except ImportError:
    print('Did not load seaborn')
    seaborn = None

__author__ = 'arbellea@post.bgu.ac.il'


def read_multi_tiff(path, start_z=None, stop_z=None, crop=None, images_out=None):
    """
    path - Path to the multipage-tiff file
    returns images stacked on axis 0
    """
    # images_out = None
    try:
        img = Image.open(path)
        if start_z is not None:
            img_itr = ImageSequence.Iterator(img)
            img_itr.position = start_z
            images_list = []
            for img_ind, img in enumerate(img_itr):
                if crop:
                    img = img.crop(crop)
                if images_out is None:
                    images_out = np.zeros((stop_z-start_z, img.size[1], img.size[0]), dtype=np.float32)
                images_out[img_ind] = np.array(img)
                if img_ind + start_z + 1 == stop_z:
                    images_out = images_out[:img_ind+1]
                    break

        else:
            images_out = np.stack([np.array(i) for i in ImageSequence.Iterator(img)], 0).astype(np.float32)
        img.close()

    finally:
        return images_out


def log_print(*args):
    now_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('{}:'.format(now_string), *args)


def get_model(model_name: str):
    try:
        model = getattr(Nets, model_name)
    except AttributeError:
        model = getattr(Nets3D, model_name)
    return model


def load_model(model_name: str, *args, **kwargs):
    model = get_model(model_name)
    if len(args) > 0 or len(**kwargs) > 0:
        return model(*args, **kwargs)
    else:
        return model


def bbox_crop(img, margin=10, three_d=False):
    # noinspection DuplicatedCode
    if three_d:
        rows = np.any(img, axis=(1, 2))
        cols = np.any(img, axis=(0, 2))
        depth = np.any(img, axis=(0, 1))
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        rmin = max(0, rmin - margin)
        cmin = max(0, cmin - margin)
        rmax = min(img.shape[0], rmax + margin)
        cmax = min(img.shape[1], cmax + margin)

        dmin, dmax = np.where(depth)[0][[0, -1]]
        dmin = max(0, dmin - margin)
        dmax = min(img.shape[2], dmax + margin)
        crop = img[rmin:rmax, cmin:cmax, dmin:dmax]
        return crop, (rmin, rmax, cmin, cmax, dmin, dmax)
    else:
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        rmin = max(0, rmin - margin)
        cmin = max(0, cmin - margin)
        rmax = min(img.shape[0], rmax + margin)
        cmax = min(img.shape[1], cmax + margin)
        crop = img[rmin:rmax, cmin:cmax]

        return crop, (rmin, rmax, cmin, cmax)


def bbox_fill(img, crop, loc, three_d=False):
    img = img.copy()
    if three_d:
        rmin, rmax, cmin, cmax, dmin, dmax = loc
        img[rmin:rmax, cmin:cmax, dmin:dmax] = crop
    else:
        rmin, rmax, cmin, cmax = loc
        img[rmin:rmax, cmin:cmax] = crop
    return img


def format_exception(e):
    if isinstance(e, Exception):
        exception_list = traceback.format_stack()
        exception_list = exception_list[:-2]
        exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
        exception_list.extend(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))

        exception_str = "Traceback (most recent call last):\n"
        exception_str += "".join(exception_list)
        # Removing the last \n
        exception_str = exception_str[:-1]

        return exception_str


def draw_labeled_on_img(img, labeled, is_coutour=False, thickness=2, im_min=None, im_max=None):
    if not img.dtype == np.uint8:
        if im_min is None:
            im_min = img.min()
        if im_max is None:
            im_max = img.max()
        img = (img - im_min) / (im_max - im_min)

    if not is_coutour:
        half_thickness = np.floor(thickness/2).astype(np.uint8)
        half_thickness_ceil = np.ceil(thickness/2).astype(np.uint8)
        kernel_erode = np.zeros((2*half_thickness+1, 2*half_thickness+1))
        kernel_dilatr = np.zeros((2*half_thickness_ceil+1, 2*half_thickness_ceil+1))
        labeled_erode = grey_erosion(labeled, structure=kernel_erode)
        labeled_dilate = grey_dilation(labeled, structure=kernel_dilatr)
        labeled_contour = labeled_dilate-labeled_erode
    else:
        labeled_contour = labeled
    img_rgb = np.stack([img]*3, axis=2)
    colors = get_random_color_pallete(0)
    new_labeled_contour = np.mod(labeled_contour, len(colors)-1) + 1
    labeled_contour = new_labeled_contour*np.greater(labeled_contour,0).astype(np.uint16)
    for p in np.unique(labeled_contour):
        if p==0:
            continue
        img_rgb[np.equal(labeled_contour, p)] = colors[p]
    img_rgb = (img_rgb*255).astype(np.uint8)
    return img_rgb

def labels2rgb(labels):
    colors = get_random_color_pallete(0)
    new_labeled = np.mod(labels, len(colors) - 1) + 1
    new_labeled = new_labeled.astype(np.uint16)
    rgb = np.zeros(labels.shape + (3,))
    for p in np.unique(new_labeled):
        rgb[new_labeled==p] = np.concatenate((colors[p], np.array([0.5])))

    rgb[np.equal(labels, 0)] = np.array([0, 0, 0, 0])
    return rgb

def get_random_color_pallete(seed=None):
    if seed is not None:
        np.random.seed(seed)
    colors = list(seaborn.color_palette('bright'))
    colors.extend(list(seaborn.color_palette('dark')))
    colors.extend(list(seaborn.color_palette('pastel')))
    colors.extend(list(seaborn.color_palette('colorblind')))
    colors.extend(list(seaborn.color_palette('muted')))
    colors.extend(list(seaborn.color_palette('deep')))
    np.random.shuffle(np.array(colors))
    return colors

