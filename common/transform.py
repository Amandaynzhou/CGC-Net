import torch
import numpy as np
import random
import torchvision.transforms.functional as tf
from scipy.spatial.distance import euclidean
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import os
import cv2
from PIL import Image, ImageOps,ImageCms
class FixRatioPoints(object):
    r"""Samples a fixed ratio of :obj:`num` points and features from a point
    cloud.

    Args:
        num (int): The number of points to sample.
    """

    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_sample = int(data.num_nodes * self.ratio)
        choice = np.random.choice(data.num_nodes, num_sample, replace=True)

        for key, item in data:
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                data[key] = item[choice]

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.ratio)



def elastic_transform(image, alpha, sigma, seed):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    image =  np.array(image)

    if len(image.shape) == 2:


        shape = image.shape

        dx = gaussian_filter((  seed), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((seed), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        image = map_coordinates(image, indices, order=1).reshape(shape)
        image = Image.fromarray(image)
        return image
    else:
        slices = []
        for i in range(3):

            aslice = image[...,i]

            shape = aslice.shape

            dx = gaussian_filter((seed), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((seed), sigma, mode="constant", cval=0) * alpha

            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

            aslice = map_coordinates(aslice, indices, order=1).reshape(shape)
            slices.append(aslice)
        slices = np.stack(slices,-1)
        slices = Image.fromarray(slices)
        return slices

class Elastic(object):
    def __init__(self, alpha, sigma=7 ,random_state = None):
        self.alpha = alpha
        self.sigma = sigma
        self.random_state = random_state
    def __call__(self , img, mask = None):
        random_state = np.random.RandomState(None)
        mask = np.array(mask)
        shape= mask.shape
        seed = random_state.rand(*shape) * 2 - 1


        return elastic_transform(img,self.alpha,self.sigma,  seed),mask
class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask = None):
        if mask is not None:
            assert img.size == mask.size
            return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask
        else:
            return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma))

class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask = None):
        if mask is not None:
            assert img.size == mask.size
            return tf.adjust_saturation(img,
                                        random.uniform(1 - self.saturation,
                                                       1 + self.saturation)), mask
        else:
            return tf.adjust_saturation(img,
                                        random.uniform(1 - self.saturation,
                                                       1 + self.saturation))

class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask=None):
        if mask is not None:
            assert img.size == mask.size
            return tf.adjust_hue(img, random.uniform(-self.hue,
                                                      self.hue)), mask
        else:
            return tf.adjust_hue(img, random.uniform(-self.hue,
                                                      self.hue))

class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask=None):
        if mask is not None:
            assert img.size == mask.size
            return tf.adjust_brightness(img,
                                        random.uniform(1 - self.bf,
                                                       1 + self.bf)), mask
        return tf.adjust_brightness(img,
                                        random.uniform(1 - self.bf,
                                                       1 + self.bf))
class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask =None):
        if mask is not None:
            assert img.size == mask.size
            return tf.adjust_contrast(img,
                                      random.uniform(1 - self.cf,
                                                     1 + self.cf)), mask
        return tf.adjust_contrast(img,
                                  random.uniform(1 - self.cf,
                                                 1 + self.cf))


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''


    def __init__(self, prob):
        self.prob = prob
        self.erasor = self.get_random_eraser()
    def get_random_eraser(self,  s_l=0.0001, s_h=0.001, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=True):
        p = self.prob
        def eraser(input_img):
            p_1 = np.random.rand()

            if p_1 > p:
                return input_img

            img_h, img_w, img_c = input_img.shape


            while True:
                s = np.random.uniform(s_l, s_h) * img_h * img_w
                r = np.random.uniform(r_1, r_2)
                w = int(np.sqrt(s / r))
                h = int(np.sqrt(s * r))
                left = np.random.randint(0, img_w)
                top = np.random.randint(0, img_h)

                if left + w <= img_w and top + h <= img_h:
                    break

            if pixel_level:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            else:
                c = np.random.uniform(v_l, v_h)

            input_img[top:top + h, left:left + w, :] = c

            return input_img

        return eraser

    def __call__(self, img,msk = None):

        num = random.randint(0, 10)
        img = np.array(img)
        for _ in range(num):
            img = self.erasor(img)
        img = Image.fromarray(img, mode="RGB")
        if msk is not None:
            return img,msk
        else:
            return img

key2aug = {'gamma': AdjustGamma,
           'hue': AdjustHue,
           'brightness': AdjustBrightness,
           'saturation': AdjustSaturation,
           'contrast': AdjustContrast,
           'elastic': Elastic,
           'erase': RandomErasing,
}

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask=None):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            if mask is not None:
                mask = Image.fromarray(mask, mode="L")
            self.PIL2Numpy = True
        if mask is not None:
            assert img.size == mask.size
            for a in self.augmentations:
                img, mask = a(img, mask)

            if self.PIL2Numpy:
                img, mask = np.array(img), np.array(mask, dtype=np.uint16)
            return img, mask
        else:
            for a in self.augmentations:

                img= a(img)

            if self.PIL2Numpy:
                img = np.array(img)
                return img

def get_composed_augmentations(aug_dict):


    augmentations = []
    if 'scale' in aug_dict.keys():
        augmentations.append(key2aug['scale'](aug_dict['scale']))
        aug_dict.pop('scale')

    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](aug_param))

    return Compose(augmentations)

if __name__ == '__main__':
    #'gamma':0.1,'hue':0.1,'brightness':0.1,'contrast':0.1,
    aug_dict={'gamma':0.1,'hue':0.1,'brightness':0.1,'contrast':0.1 }
    augments = get_composed_augmentations(aug_dict)
    filepath  = '/media/amanda/HDD2T_1/warwick-research/data/raw/ICIAR/train_aug'
    files = os.listdir(filepath)
    rawfiles = [os.path.join(filepath,f) for f in files]
    maskfiles = [os.path.join('/media/amanda/HDD2T_1/warwick-research/data/proto/mask/ICIAR/train_aug',f.replace('.tif','.npy')  ) for f in files]
    for raw,mask in zip(rawfiles,maskfiles):
        img = cv2.imread(raw)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # msk =np.load(mask)
        img = augments(img,None)
        cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.imwrite(raw.replace('train_aug/','train_aug3/c2_'), img)
        # cv2.imwrite(mask.replace('train','train_aug'),msk)

