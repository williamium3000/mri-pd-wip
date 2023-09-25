import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy import ndimage
import random
from torchvision.transforms import functional as F
import torch

class PairedMinMaxNormalize():
    """Apply min max normalization to image
    """
    def __call__(self, input):
        image1, image2 = input[0], input[1]
        image1 = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
        image2 = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))
        return (image1, image2)
    
class Identity():
    """Apply min max normalization to image
    """
    def __call__(self, input):
        return input
    
class PairedStdNormalize():
    """Normalize each image to with mean and std.

    Args:
        mean (tuple | None): mean of the image.
        std (tuple | None): std of the image.
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, input):
        image1, image2 = input[0], input[1]
        c = image1.shape[-1]
        if self.mean is None:
            mean1 = np.mean(image1.reshape(-1, c), axis=0)
            mean2 = np.mean(image2.reshape(-1, c), axis=0)
        else:
            mean1 = np.array(self.mean)
            mean2 = np.array(self.mean)
        if self.std is None:
            std1 = np.std(image1.reshape(-1, c), axis=0)
            std2 = np.std(image2.reshape(-1, c), axis=0)
        else:
            std1 = np.array(self.std)
            std2 = np.array(self.std)
        
        image1 = (image1 - mean1) / std1
        image2 = (image2 - mean2) / std2
        return (image1, image2)
    
class PairedNoneZeroRegion3D():
    """Extract none-zero region. This only works on 3D images (4D tensor)
    """
    def __call__(self, input):
        image1, image2 = input[0], input[1]
        assert (len(image1.shape) == 4) and (len(image2.shape) == 4), "ExtractRegion3D only works on 3D images (4D tensor)"
        
        nonzero_mask = np.zeros(image1.shape[:-1], dtype=bool)
        for c in range(image1.shape[-1]):
            mask_c1 = image1[:, :, :, c] != 0
            mask_c2 = image2[:, :, :, c] != 0
            nonzero_mask = nonzero_mask | mask_c1 | mask_c2
            
        nonzero_mask = binary_fill_holes(nonzero_mask)

        mask_voxel_coords = np.where(nonzero_mask != 0)
        minxidx = int(np.min(mask_voxel_coords[0]))
        maxxidx = int(np.max(mask_voxel_coords[0])) + 1
        minyidx = int(np.min(mask_voxel_coords[1]))
        maxyidx = int(np.max(mask_voxel_coords[1])) + 1
        minzidx = int(np.min(mask_voxel_coords[2]))
        maxzidx = int(np.max(mask_voxel_coords[2])) + 1
        bbox = [[minxidx, maxxidx], [minyidx, maxyidx], [minzidx, maxzidx]]

        image1 = image1[
            bbox[0][0]: bbox[0][1],
            bbox[1][0]: bbox[1][1],
            bbox[2][0]: bbox[2][1],
            :
        ]

        image2 = image2[
            bbox[0][0]: bbox[0][1],
            bbox[1][0]: bbox[1][1],
            bbox[2][0]: bbox[2][1]
        ]
        
        return (image1, image2)

class PairedRandomCrop3D():
    """Random Crop for 3D medical image. This transform makes
    sure that the output is absolutely in the size of the crop_size
    given. If any dimension of the input image is less than crop_size,
    this transform will first pad the image to given crop_size

    Args:
        crop_size (tuple | None): crop size.
    """

    def __init__(self, crop_size, pad_val=0):
        if isinstance(crop_size, int):
            crop_size = [crop_size] * 3
        self.crop_size = crop_size
        self.pad_val = pad_val

    def __call__(self, input):
        image1, image2 = input[0], input[1]
        crop_h, crop_w, crop_z = self.crop_size
        h, w, z, _ = image1.shape
        # first pad to sizes larger than crop size
        image1 = np.pad(
            image1,
            [(0, max(crop_h - h, 0)), (0, max(crop_w - w, 0)), (0, max(crop_z - z, 0)), (0, 0)],
            mode='constant',
            constant_values=self.pad_val)
        image2 = np.pad(
            image2,
            [(0, max(crop_h - h, 0)), (0, max(crop_w - w, 0)), (0, max(crop_z - z, 0)), (0, 0)],
            mode='constant',
            constant_values=self.pad_val)
        # random crop to crop size
        h, w, z, _ = image1.shape
        sagittal = np.random.randint(0, h - crop_h + 1)
        coronal = np.random.randint(0, w - crop_w + 1)
        axial = np.random.randint(0, z - crop_z + 1)
        
        image1 = image1[
            sagittal:sagittal + crop_h,
            coronal:coronal + crop_w,
            axial:axial + crop_z,
            :
        ]
        image2 = image2[
            sagittal:sagittal + crop_h,
            coronal:coronal + crop_w,
            axial:axial + crop_z,
            :
        ]
        return (image1, image2)
    

class PairedRandomRotation3d():
    """Rotate the image for a random degree.
    """

    def __init__(self, prob, degree=(20, -20), plane=None):
        self.prob = prob
        self.plane = plane
        self.degree = degree
        if not isinstance(degree, (tuple, list)):
            self.degree = (degree, -degree)
    def __call__(self, input):
        image1, image2 = input[0], input[1]
        if np.random.rand() < self.prob:
            if self.plane is None:
                plane = random.choice([(0, 1), (1, 2), (0, 2)])
            else:
                plane = self.plane
            angle = random.uniform(a=self.degree[0], b=self.degree[1])
            image1 = ndimage.rotate(image1, angle, axes=plane, reshape=False, order=0)
            image2 = ndimage.rotate(image2, angle, axes=plane, reshape=False, order=0)
        return (image1, image2)
    
     
class PairedRandomRotation90n3d():
    """Random rotate the image 90 degree for a maximum of k times
    (randomly choose the number of times).
    """

    def __init__(self, prob, max_k=4, plane=None):
        self.prob = prob
        self.plane = plane
        self.max_k = max_k
    def __call__(self, input):
        image1, image2 = input[0], input[1]
        if np.random.rand() < self.prob:
            if self.plane is None:
                plane = random.choice([(0, 1), (1, 2), (0, 2)])
            else:
                plane = self.plane
            k = np.random.randint(0, self.max_k)
            image1 = np.rot90(image1, k=k, axes=plane)
            image2 = np.rot90(image2, k=k, axes=plane,)
        return (image1, image2)
    


class PairedPad3D(object):
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

    def __call__(self, input):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        image1, image2 = input[0], input[1]
        padded_image1 = self._pad_img(image1)
        padded_image2 = self._pad_img(image2)

        return (padded_image1, padded_image2)


class PairedRandomFlip3D():
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
        image1, image2 = input[0], input[1]
        if flip_direction == 'horizontal':
            axis = 1
        elif flip_direction == 'vertical':
            axis = 0
        elif flip_direction == 'inferior':
            axis = 2
        
        image1 = np.flip(image1, axis=axis).copy()
        image2 = np.flip(image2, axis=axis).copy()
        return (image1, image2)


class PairedToTensor3d:
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
        image1, image2 = input[0], input[1]
        image1 = torch.from_numpy(image1).permute(3, 0, 1, 2).float().contiguous()
        image2 = torch.from_numpy(image2).permute(3, 0, 1, 2).float().contiguous()
        return (image1, image2)