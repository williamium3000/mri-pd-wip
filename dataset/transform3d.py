import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy import ndimage
import random
from torchvision.transforms import functional as F
import torch

class MinMaxNormalize():
    """Apply min max normalization to image
    """
    def __call__(self, input):
        image, mask = input[0], input[1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        return (image, mask)
class Identity():
    """Apply min max normalization to image
    """
    def __call__(self, input):
        return input
    
class StdNormalize():
    """Normalize each image to with mean and std.

    Args:
        mean (tuple | None): mean of the image.
        std (tuple | None): std of the image.
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, input):
        image, mask = input[0], input[1]
        c = image.shape[-1]
        if self.mean is None:
            mean = np.mean(image.reshape(-1, c), axis=0)
        else:
            mean = np.array(mean)
        if self.std is None:
            std = np.std(image.reshape(-1, c), axis=0)
        else:
            std = np.array(std)
        
        image = (image - mean) / std
        return (image, mask)
class NoneZeroRegion3D():
    """Extract none-zero region. This only works on 3D images (4D tensor)
    """
    def __call__(self, input):
        image, mask = input[0], input[1]
        assert len(image.shape) == 4, "ExtractRegion3D only works on 3D images (4D tensor)"
        
        nonzero_mask = np.zeros(image.shape[:-1], dtype=bool)
        for c in range(image.shape[-1]):
            mask_c = image[:, :, :, c] != 0
            nonzero_mask = nonzero_mask | mask_c
        nonzero_mask = binary_fill_holes(nonzero_mask)

        mask_voxel_coords = np.where(nonzero_mask != 0)
        minxidx = int(np.min(mask_voxel_coords[0]))
        maxxidx = int(np.max(mask_voxel_coords[0])) + 1
        minyidx = int(np.min(mask_voxel_coords[1]))
        maxyidx = int(np.max(mask_voxel_coords[1])) + 1
        minzidx = int(np.min(mask_voxel_coords[2]))
        maxzidx = int(np.max(mask_voxel_coords[2])) + 1
        bbox = [[minxidx, maxxidx], [minyidx, maxyidx], [minzidx, maxzidx]]

        image = image[
            bbox[0][0]: bbox[0][1],
            bbox[1][0]: bbox[1][1],
            bbox[2][0]: bbox[2][1],
            :
        ]

        mask = mask[
            bbox[0][0]: bbox[0][1],
            bbox[1][0]: bbox[1][1],
            bbox[2][0]: bbox[2][1]
        ]
        
        return (image, mask)

class RemoveBackgroundSlices():
    """Remove slices contraining background label. This only works on 3D images (4D tensor)

    """

    def __init__(self, id=0):
        self.id = id

    def __call__(self, input):
        image, mask = input[0], input[1]
        assert len(image.shape) == 4, "RemoveBackgroundSlices only works on 3D images (4D tensor)"
        
        # drop slices without any label
        z_mask = np.any(mask != self.id, axis=(0, 1))  
              
        image = image[:, :, z_mask, :]
        mask = mask[:, :, z_mask]

        return (image, mask)

class RandomCrop3D():
    """Random Crop for 3D medical image. This transform makes
    sure that the output is absolutely in the size of the crop_size
    given. If any dimension of the input image is less than crop_size,
    this transform will first pad the image to given crop_size

    Args:
        crop_size (tuple | None): crop size.
    """

    def __init__(self, crop_size, pad_val=0, seg_pad_val=255):
        if isinstance(crop_size, int):
            crop_size = [crop_size] * 3
        self.crop_size = crop_size
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def __call__(self, input):
        image, mask = input[0], input[1]
        crop_h, crop_w, crop_z = self.crop_size
        h, w, z, _ = image.shape
        # first pad to sizes larger than crop size
        image = np.pad(
            image,
            [(0, max(crop_h - h, 0)), (0, max(crop_w - w, 0)), (0, max(crop_z - z, 0)), (0, 0)],
            mode='constant',
            constant_values=self.pad_val)
        mask = np.pad(
            mask,
            [(0, max(crop_h - h, 0)), (0, max(crop_w - w, 0)), (0, max(crop_z - z, 0))],
            mode='constant',
            constant_values=self.seg_pad_val)
        
        # random crop to crop size
        h, w, z, _ = image.shape
        sagittal = np.random.randint(0, h - crop_h + 1)
        coronal = np.random.randint(0, w - crop_w + 1)
        axial = np.random.randint(0, z - crop_z + 1)
        
        image = image[
            sagittal:sagittal + crop_h,
            coronal:coronal + crop_w,
            axial:axial + crop_z,
            :
        ]
        mask = mask[
            sagittal:sagittal + crop_h,
            coronal:coronal + crop_w,
            axial:axial + crop_z
        ]

        return (image, mask)
    

class RandomRotation3d():
    """Rotate the image for a random degree.
    """

    def __init__(self, prob, degree=(20, -20), plane=None):
        self.prob = prob
        self.plane = plane
        self.degree = degree
        if not isinstance(degree, (tuple, list)):
            self.degree = (degree, -degree)
    def __call__(self, input):
        image, mask = input[0], input[1]
        if np.random.rand() < self.prob:
            if self.plane is None:
                plane = random.choice([(0, 1), (1, 2), (0, 2)])
            else:
                plane = self.plane
            angle = random.uniform(a=self.degree[0], b=self.degree[1])
            image = ndimage.rotate(image, angle, axes=plane, reshape=False, order=0)
            mask = ndimage.rotate(mask, angle, axes=plane, reshape=False, order=0)
        return (image, mask)
    
     
class RandomRotation90n3d():
    """Random rotate the image 90 degree for a maximum of k times
    (randomly choose the number of times).
    """

    def __init__(self, prob, max_k=4, plane=None):
        self.prob = prob
        self.plane = plane
        self.max_k = max_k
    def __call__(self, input):
        image, mask = input[0], input[1]
        if np.random.rand() < self.prob:
            if self.plane is None:
                plane = random.choice([(0, 1), (1, 2), (0, 2)])
            else:
                plane = self.plane
            k = np.random.randint(0, self.max_k)
            image = np.rot90(image, k=k, axes=plane)
            mask = np.rot90(mask, k=k, axes=plane,)
        return (image, mask)
    

class Pad3D(object):
    """Pad the image & mask for 3D input to given size or the
    minimum size that is divisible by some number. If any dimension of the 
    3D image is already larger than given pad size, this dimension will 
    be ignored (no padding and no cropping).
    Added keys are "pad_shape", "pad_fixed_size".

    Args:
        size (tuple, optional): Fixed padding size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
        assert size is None or size_divisor is None
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val


    def _pad_img(self, img):
        """Pad images according to ``self.size``."""

        h, w, z, _ = img.shape
        if self.size is not None:
            padded_img = np.pad(
                img,
                [(0, max(self.size[0] - h, 0)), (0, max(self.size[1] - w, 0)), (0, max(self.size[2] - z, 0)), (0, 0)],
                mode='constant',
                constant_values=self.pad_val)
        elif self.size_divisor is not None:
            pad_h = int(np.ceil(h / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(w / self.size_divisor)) * self.size_divisor
            pad_z = int(np.ceil(z / self.size_divisor)) * self.size_divisor
            padded_img = np.pad(
                img,
                [(0, max(pad_h - h, 0)), (0, max(pad_w - w, 0)), (0, max(pad_z - z, 0)), (0, 0)],
                mode='constant',
                constant_values=self.pad_val)
        return padded_img

    def _pad_seg(self, padded_img, mask):
        """Pad masks according to ``results['pad_shape']``."""
        h, w, z = mask.shape
        pad_h, pad_w, pad_z, _ = padded_img.shape
        padded_mask = np.pad(
            mask,
            [(0, max(pad_h - h, 0)), (0, max(pad_w - w, 0)), (0, max(pad_z - z, 0))],
            mode='constant',
            constant_values=self.seg_pad_val
        )
        return padded_mask

    def __call__(self, input):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        image, mask = input[0], input[1]
        padded_image = self._pad_img(image)
        padded_mask = self._pad_seg(padded_image, mask)
        return (padded_image, padded_mask)


class RandomFlip3D():
    """Random flip for 3D image.

    Args:
        direction (str | (horizontal, vertical, inferior)): flip direction.
    """

    def __init__(self, prob, direction=None):
        self.prob = prob
        self.direction = direction
        if direction is not None:
            assert direction in ['horizontal', 'vertical', 'inferior']
    def __call__(self, input):
        if np.random.rand() < self.prob:
            if self.direction is not None:
                flip_direction = self.direction
            else:
                flip_direction = \
                    ['horizontal', 'vertical', 'inferior'][np.random.randint(0, 3)]
            input = self.flip_image(input, flip_direction)
            
        return input
    def flip_image(self, input, flip_direction):
        image, mask = input[0], input[1]
        if flip_direction == 'horizontal':
            axis = 1
        elif flip_direction == 'vertical':
            axis = 0
        elif flip_direction == 'inferior':
            axis = 2
        
        image = np.flip(image, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()
        return (image, mask)


class ToTensor3d:
    """Convert a 3D ``numpy.ndarray`` to tensor (both image and mask).

    Converts a 3D image of numpy.ndarray (H x W x Z x C) to a torch.FloatTensor of 
    shape (C x H x W x Z). 

    Converts a 3D mask of numpy.ndarray (H x W x Z) to a torch.FloatTensor of 
    shape (H x W x Z). 
    
    """

    def __call__(self, input):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        image, mask = input[0], input[1]
        image = torch.from_numpy(image.transpose((3, 0, 1, 2))).float().contiguous()
        mask = torch.from_numpy(mask.copy()).long()
        return (image, mask)