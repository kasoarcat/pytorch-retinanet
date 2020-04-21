import albumentations as albu
from albumentations.pytorch.transforms import ToTensor
import torch
import numpy as np
import cv2

def get_augumentation(phase, width=512, height=512, min_area=0., min_visibility=0.):
    list_transforms = []
    if phase == 'train':
        list_transforms.extend([
            albu.augmentations.transforms.LongestMaxSize(max_size=width, always_apply=True),
            albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0, value=[0, 0, 0]),
            albu.augmentations.transforms.RandomResizedCrop(height=height, width=width, p=0.3),
            albu.augmentations.transforms.Flip(),
            albu.augmentations.transforms.Transpose(),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.4),
                albu.RandomGamma(gamma_limit=(50, 150)),
                albu.NoOp()
            ]),
            albu.OneOf([
                albu.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
                albu.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5),
                albu.NoOp()
            ]),
            albu.CLAHE(p=0.8),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
        ])
    if(phase == 'test' or phase == 'valid'):
        list_transforms.extend([
            albu.Resize(height=height, width=width)
        ])
    list_transforms.extend([
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ToTensor()
    ])
    if(phase == 'test'):
        return albu.Compose(list_transforms)
    return albu.Compose(list_transforms, bbox_params=albu.BboxParams(format='coco', min_area=min_area,
                                                                     min_visibility=min_visibility, 
                                                                     label_fields=['category_id']))
    