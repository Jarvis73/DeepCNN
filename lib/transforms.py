# Copyright 2019 Jianwei Zhang All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =================================================================================

"""
Here we provide some basic image transformation operations implemented
by Numpy and Tensorflow.

"""

import cv2
import math
import random
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.contrib.image import transform


__all__ = [
    "Compose", "Standardize", "Normalize", "Pad",
    "RandomHorizontalFlip", "RandomVerticalFlip", "RandomCrop", "RandomAffine"
]


def _is_tensor(x):
    """Returns `True` if `x` is a symbolic tensor-like object.

    Args:
        x: A python object to check.

    Returns:
        `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
    """
    return isinstance(x, (ops.Tensor, variables.Variable))


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms, tensor=False):
        self.transforms = transforms
        self.tensor = tensor

    def __call__(self, image):
        for t in self.transforms:
            image = t(image, tensor=self.tensor)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Standardize(object):
    def __init__(self, scale=255.):
        self.scale = scale

    def __call__(self, image, **kwargs):
        if kwargs.get("tensor", False):
            with tf.name_scope(kwargs.get("name", self.__class__.__name__.lower())):
                return tf.cast(image, dtype=tf.float32) / self.scale
        else:
            return np.asarray(image, dtype=np.float32) / self.scale

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)[None, None, :]
        self.std = np.asarray(std, dtype=np.float32)[None, None, :]
        if np.any(self.std == 0.):
            warnings.warn("There exists zeros in 'std' parameters which maybe cause "
                          "nan in 'Normalize' operation!")

    def __call__(self, image, **kwargs):
        if kwargs.get("tensor", False):
            with tf.name_scope(kwargs.get("name", self.__class__.__name__.lower())):
                return tf.divide(tf.subtract(image, self.mean), self.std)
        else:
            return (image - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Pad(object):
    def __init__(self, padding, fill=0, padding_mode="constant"):
        assert padding_mode in ["constant", "reflect", "symmetric"]
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        if isinstance(self.padding, int):
            self.padding = [[self.padding] * 2, [self.padding] * 2, [0, 0]]
        elif isinstance(self.padding, (tuple, list)) and len(self.padding) == 2:
            self.padding = [[self.padding[0]] * 2, [self.padding[1]] * 2, [0, 0]]

    def __call__(self, image, **kwargs):
        if kwargs.get("tensor", False):
            with tf.name_scope(kwargs.get("name", self.__class__.__name__.lower())):
                return tf.pad(image, self.padding, self.padding_mode, constant_values=self.fill)
        else:
            if self.padding_mode == "constant":
                return np.pad(image, self.padding, self.padding_mode, constant_values=self.fill)
            else:
                return np.pad(image, self.padding, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.padding, self.fill, self.padding_mode)


class RandomHorizontalFlip(object):
    def __call__(self, image, **kwargs):
        seed = kwargs.get("seed", None)
        if kwargs.get("tensor", False):
            with tf.name_scope(kwargs.get("name", self.__class__.__name__.lower())):
                return tf.image.random_flip_left_right(image, seed)
        else:
            print(image.shape)
            if random.random() < .5:
                return np.fliplr(image)
            return image

    def __repr__(self):
        return self.__class__.__name__ + '(p=0.5)'


class RandomVerticalFlip(object):
    def __call__(self, image, **kwargs):
        seed = kwargs.get("seed", None)
        if kwargs.get("tensor", False):
            with tf.name_scope(kwargs.get("name", self.__class__.__name__.lower())):
                return tf.image.random_flip_up_down(image, seed)
        else:
            if random.random() < .5:
                return np.flipud(image)
            return image

    def __repr__(self):
        return self.__class__.__name__ + '(p=0.5)'


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant", reflect_type="even"):
        if isinstance(size, int):
            self.size = [size, size]
        else:
            self.size = size
        assert len(self.size) == 2, "'size' must be a sequence with two elements"
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.reflect_type = reflect_type
        if isinstance(self.padding, int):
            self.padding = [[self.padding] * 2, [self.padding] * 2, [0, 0]]
        elif isinstance(self.padding, (tuple, list)) and len(self.padding) == 2:
            self.padding = [[self.padding[0]] * 2, [self.padding[1]] * 2, [0, 0]]

    def __call__(self, image, **kwargs):
        """ This implementation is referenced from PyTorch.
        """
        seed = kwargs.get("seed", None)
        if kwargs.get("tensor", False):
            with tf.name_scope(kwargs.get("name", self.__class__.__name__.lower())):
                if self.padding is not None:
                    image = tf.pad(image, self.padding, self.padding_mode, constant_values=self.fill)
                height, width, channel = self._image_dimensions(image, rank=3)
                if self.pad_if_needed:
                    offset_pad_height = self.max_(self.size[0] - height, 0)
                    offset_pad_width = self.max_(self.size[1] - width, 0)
                    image = tf.image.pad_to_bounding_box(
                        image, offset_pad_height, offset_pad_width,
                        self.size[0] + offset_pad_height, self.size[1] + offset_pad_width)
                return tf.image.random_crop(image, tf.concat((self.size, [channel]), axis=0), seed)
        else:
            if self.padding is not None:
                image = np.pad(image, self.padding, self.padding_mode, constant_values=self.fill)
            if self.pad_if_needed:
                height, width, _ = image.shape
                pad_height = max(self.size[0] - height, 0)
                pad_width = max(self.size[1] - width, 0)
                if self.padding_mode == "constant":
                    image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)),
                                   mode=self.padding_mode, constant_values=self.fill)
                else:
                    image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)),
                                   mode=self.padding_mode, reflect_type=self.reflect_type)
            i, j, h, w = self.get_params(image.shape[:-1], self.size)
            return image[i:i + h, j:j + w]

    @staticmethod
    def get_params(input_size, output_size):
        w, h = input_size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    @staticmethod
    def _image_dimensions(image, rank):
        """Returns the dimensions of an image tensor.

        Args:
          image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
          rank: The expected rank of the image

        Returns:
          A list of corresponding to the dimensions of the
          input image.  Dimensions that are statically known are python integers,
          otherwise they are integer scalar tensors.
        """
        if image.get_shape().is_fully_defined():
            return image.get_shape().as_list()
        else:
            static_shape = image.get_shape().with_rank(rank).as_list()
            dynamic_shape = tf.unstack(tf.shape(image), rank)
            return [
                s if s is not None else d for s, d in zip(static_shape, dynamic_shape)
            ]

    @staticmethod
    def max_(x, y):
        if _is_tensor(x) or _is_tensor(y):
            return tf.math.maximum(x, y)
        else:
            return max(x, y)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Notation: This class is copied or modified from PyTorch.

    Parameters
    ----------
    degrees (sequence or float or int): Range of degrees to select from.
        If degrees is a number instead of sequence like (min, max), the range of degrees
        will be (-degrees, +degrees). Set to 0 to deactivate rotations.
    translate (tuple, optional): tuple of maximum absolute fraction for horizontal
        and vertical translations. For example translate=(a, b), then horizontal shift
        is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
        randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
    scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
        randomly sampled from the range a <= scale <= b. Will keep original scale by default.
    shear (sequence or float or int, optional): Range of degrees to select from.
        If degrees is a number instead of sequence like (min, max), the range of degrees
        will be (-degrees, +degrees). Will not apply shear by default
    order (choice from [0, 1, 3], optional):
        An optional resampling order.
    border (choice from [constant, reflect, reflect_101, repeat, wrap], optional):
        border type
    constant_value (int): Optional fill color for the area outside the transform in the output image.

    Notation
    --------
    Restriction: Tensorflow only support order in [0, 1] and border padding 0
    """
    def __init__(self, degrees=None, translate=None, scale=None, shear=None, order=1, border="constant", constant_value=0):
        if degrees is not None:
            if isinstance(degrees, (float, int)):
                if degrees < 0:
                    raise ValueError("If degrees is a single number, it must be positive.")
                self.degrees = (-degrees, degrees)
            else:
                assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                    "degrees should be a list or tuple and it must be of length 2."
                self.degrees = degrees
        else:
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, (float, int)):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        if order == 0:
            self.order = cv2.INTER_NEAREST
            self.order_tf = "NEAREST"
        elif order == 1:
            self.order = cv2.INTER_LINEAR
            self.order_tf = "BILINEAR"
        elif order == 3:
            self.order = cv2.INTER_CUBIC
            self.order_tf = "BILINEAR"
        else:
            raise ValueError("`order` must be choice from [0, 1, 3]")

        if border == "constant":
            self.border = cv2.BORDER_CONSTANT
        elif border == "reflect":
            self.border = cv2.BORDER_REFLECT
        elif border == "reflect_101":
            self.border = cv2.BORDER_REFLECT101
        elif border == "repeat":
            self.border = cv2.BORDER_REPLICATE
        elif border == "wrap":
            self.border = cv2.BORDER_WRAP
        else:
            raise ValueError("`border` must be choice from [constant, reflect, reflect_101, repeat, wrap]")

        self.constant = constant_value

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size, seed=None):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        random.seed(seed)
        if degrees is not None:
            angle = random.uniform(degrees[0], degrees[1])
        else:
            angle = 0
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    @staticmethod
    def get_size(image):
        if isinstance(image, np.ndarray):
            return image.shape
        elif isinstance(image, tf.Tensor):
            return image.get_shape().as_list()
        else:
            raise TypeError("`image` must be np.ndarray or tf.Tensor, got {}".format(type(image)))

    @staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RSS is rotation with scale and shear matrix
        #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
        #                              [ sin(a)*scale    cos(a + shear)*scale      0]
        #                              [     0                  0                  1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale = 1.0 / scale

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle + shear), math.sin(angle + shear), 0,
            -math.sin(angle), math.cos(angle), 0
        ]
        matrix = [scale / d * m for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]
        return matrix

    def __call__(self, image, **kwargs):
        seed = kwargs.get("seed", None)
        size = self.get_size(image)
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, size, seed)
        assert len(size) == 3, "`image` must be [h, w, c]"
        if isinstance(self.constant, int):
            self.constant = [self.constant] * size[-1]
        elif len(self.constant) != size[-1]:
            raise ValueError("`constant` value must have the same number with image channels, got {}"
                             " and image has {} channels."
                             .format(self.constant, size[-1]))
        center = (size[0] * 0.5 + 0.5, size[1] * 0.5 + 0.5)
        matrix = self._get_inverse_affine_matrix(center, *ret)
        if isinstance(image, tf.Tensor):
            matrix = matrix + [0., 0.]
            with tf.name_scope(kwargs.get("name", self.__class__.__name__.lower())):
                return transform(image, matrix, interpolation=self.order_tf, output_shape=size[:-1])
        else:
            matrix = np.array(matrix, np.float32).reshape(2, 3)
            return cv2.warpAffine(image, matrix[:2], dsize=size[1::-1], flags=self.order,
                                  borderMode=self.border, borderValue=self.constant)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        s += ', order={order}, constant={constant})'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)


ZScore = Normalize

##################################################################
#
#   Mix-up method implementation
#
##################################################################


class Mixup(object):
    def __init__(self, alpha=0.2, seed=None):
        self.alpha = alpha
        if isinstance(seed, (tuple, list)):
            self.seed1, self.seed2 = seed
        else:
            self.seed1, self.seed2 = None, None

    def __call__(self, images, labels):
        if isinstance(images, tf.Tensor):
            # Assert images is 4D Tensor
            with tf.name_scope("Mixup"):
                batch_size = images.get_shape().as_list()[0]
                if batch_size is None:
                    batch_size = tf.shape(images)[0]
                    perm = tf.random.shuffle(tf.range(batch_size), self.seed1)
                else:
                    perm = np.random.permutation(batch_size)
                shuffled_images = tf.gather(images, perm)
                labels = tf.cast(labels, tf.float32)
                shuffled_labels = tf.gather(labels, perm)
                lam = tf.distributions.Beta(self.alpha, self.alpha).sample(seed=self.seed2)
                mixed_images = lam * images + (1 - lam) * shuffled_images
                mixed_labels = lam * labels + (1 - lam) * shuffled_labels
        else:
            batch_size = images.shape[0]
            perm = np.random.permutation(batch_size)
            shuffled_images = images[perm]
            shuffled_labels = labels[perm]
            lam = np.random.beta(self.alpha, self.alpha)
            mixed_images = lam * images + (1 - lam) * shuffled_images
            mixed_labels = lam * labels + (1 - lam) * shuffled_labels
        return mixed_images, mixed_labels

    def __repr__(self):
        return self.__class__.__name__ + '(alpha={})'.format(self.alpha)
