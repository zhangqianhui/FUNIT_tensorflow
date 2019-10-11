from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import imageio
import scipy.misc as misc
import scipy
import numpy as np

def save_as_gif(images_list, out_path, gif_file_name='all', save_image=False):

    if os.path.exists(out_path) == False:
        os.mkdir(out_path)
    # save as .png
    if save_image == True:
        for n in range(len(images_list)):
            file_name = '{}.png'.format(n)
            save_path_and_name = os.path.join(out_path, file_name)
            misc.imsave(save_path_and_name, images_list[n])
    # save as .gif
    out_path_and_name = os.path.join(out_path, '{}.gif'.format(gif_file_name))
    imageio.mimsave(out_path_and_name, images_list, 'GIF', duration=0.1)

def get_image(image_path, crop_size=128, is_crop=False, resize_w=140, is_grayscale=False):
    return transform(imread(image_path , is_grayscale), crop_size, is_crop, resize_w)

def transform(image, crop_size=64, is_crop=True, resize_w=140):

    image = scipy.misc.imresize(image, [resize_w, resize_w])
    if is_crop:
        cropped_image = center_crop(image, crop_size)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])

    return np.array(cropped_image) / 127.5 - 1

def center_crop(x, crop_h, crop_w=None):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    rate = np.random.uniform(0, 1, size=1)
    if rate < 0.5:
        x = np.fliplr(x)

    return x[j:j+crop_h, i:i+crop_w]

def save_images(images, size, image_path, is_ouput=False):

    return imsave(inverse_transform(images, is_ouput), size, image_path)

def imread(path, is_grayscale=False):

    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):

    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):

    if size[0] + size[1] == 2:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    else:

        h, w = images.shape[1], images.shape[2]
        img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def inverse_transform(image, is_ouput=False):

    if is_ouput == True:
        print(image[0])
    result = ((image + 1) * 127.5).astype(np.uint8)
    if is_ouput == True:
        print(result)
    return result