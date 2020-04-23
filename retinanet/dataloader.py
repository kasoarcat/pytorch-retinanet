from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler, RandomSampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image
import h5py
import cv2

class H5CoCoDataset(torch.utils.data.Dataset):
    def __init__(self, path, set_name):
        self.path = path
        self.set_name = set_name
        self.index = 0
        with h5py.File(self.path, 'r') as f:
            self.len = len(f["img"])
            self.image_ids = [int(i) for i in f['image_ids']]
            self.coco_labels = dict(zip(f['coco_labels_keys'], f['coco_labels_values']))

    def __getitem__(self, index):
        img = None
        annot = None
        with h5py.File(self.path, 'r') as f:
            img = torch.tensor(f["img"][str(index)])
            annot = torch.tensor(f["annot"][str(index)])
            scale = np.array(f["scale"][str(index)])
        return {'img':img, 'annot':annot, 'scale':scale}
    
    def __len__(self):
        return self.len
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < self.len:
            item = self[self.index]
            self.index += 1
            return item
        else:
            self.index = 0
            raise StopIteration
            
    def image_aspect_ratio(self, image_index):
        img = self[image_index]['img']
        return float(img.shape[0]) / float(img.shape[1])

    def label_to_coco_label(self, label):
        return int(self.coco_labels[label])


class CocoDataset(Dataset):
    def __init__(self, root_dir, set_name='train2017', do_aug=0, transform=None, limit_len=0):
        self.root_dir = root_dir
        self.set_name = set_name
        self.do_aug = do_aug
        self.transform = transform
        self.coco = COCO(os.path.join(self.root_dir, self.set_name, self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.load_classes()
        if limit_len == 0:
            self.len = len(self.image_ids)
        else:
            self.len = limit_len
        self.index = 0

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __getitem__(self, idx):
        # print('idx:', idx)
        sample = None
        if not self.do_aug:
            img = self.load_image(idx)
            # print('img:', img.shape)
            annot = self.load_annotations(idx)
            sample = {'img': img, 'annot': annot}
            if self.transform:
                sample = self.transform(sample)
        else:
            sample = {'bboxes': [], 'category_id': [], 'image': self.load_image(idx)}
            annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx], iscrowd=False)
            if len(annotations_ids) == 0:
                # print('annotations_ids == 0 idx:{} image_ids:{}'.format(idx, self.image_ids[idx]))
                pass
                # sample['bboxes'] = np.zeros((0, 4))
                # sample['category_id'] = np.zeros((0, 1))
            else:
                coco_annotations = self.coco.loadAnns(annotations_ids)
                for idx, a in enumerate(coco_annotations):
                    sample['bboxes'].append(a['bbox'])
                    sample['category_id'].append(self.coco_label_to_label(a['category_id']))

            if self.transform:
                sample = self.transform(**sample)
            
            # transform from [x, y, w, h] to [x1, y1, x2, y2]
            sample['bboxes'] = [list(i) for i in sample["bboxes"]]
            sample['bboxes'] = np.array(sample['bboxes'])
            if len(sample['bboxes']) > 0:
                sample['bboxes'][:, 2] = sample['bboxes'][:, 0] + sample['bboxes'][:, 2]
                sample['bboxes'][:, 3] = sample['bboxes'][:, 1] + sample['bboxes'][:, 3]
            # print('bbox:', sample['bboxes'])

        return sample

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.len:
            item = self[self.index]
            self.index += 1
            return item
        else:
            self.index = 0
            raise StopIteration

    # def __add__(self, other):
    #     return CocoDataset([self, other])

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, 'images', image_info['file_name'])
        img = skimage.io.imread(path)
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        img = img.astype(np.float32) / 255.0
        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])


def detection_collate(batch):
    imgs = [s['image'] for s in batch]
    annots = [s['bboxes'] for s in batch]
    labels = [s['category_id'] for s in batch]
    # # print('imgs:', len(imgs))
    # # print('annots:', len(annots))
    # # print('labels:', len(labels))

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)
    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = torch.FloatTensor(img)

    # max_num_annots = max(len(annot) for annot in annots)
    # annot_padded = np.ones((len(annots), max_num_annots, 5)) * -1
    # if max_num_annots > 0:
    #     for idx, (annot, lab) in enumerate(zip(annots, labels)):
    #         if len(annot) > 0:
    #             annot_padded[idx, :len(annot), :4] = annot
    #             annot_padded[idx, :len(annot), 4] = lab
    # return {'img': padded_imgs, 'annot': torch.FloatTensor(annot_padded)}

    max_num_annots = max(len(annot) for annot in annots)
    annot_padded = np.ones((len(annots), max_num_annots, 5)) * -1
    if max_num_annots > 0:
        for idx, (annot, lab) in enumerate(zip(annots, labels)):
            if len(annot) > 0:
                annot_padded[idx, :len(annot), :4] = annot
                annot_padded[idx, :len(annot), 4] = lab
    
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    return {'img': padded_imgs, 'annot': torch.FloatTensor(annot_padded)}


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)
    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    def __init__(self, min_side, max_side):
        self.min_side = min_side
        self.max_side = max_side

    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # print('Resizer __call__')
        image, annots = sample['img'], sample['annot']
        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = self.min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, flip_x=0.5):
        # print('Augmenter __call__')
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}
        return sample


class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        # print('Normalizer __init__')
        image, annots = sample['img'], sample['annot']
        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_sources, batch_size, drop_last):
        self.data_sources = data_sources
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_sources) // self.batch_size
        else:
            return (len(self.data_sources) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_sources)))
        # print('len:', len(order))

        if type(self.data_sources) == Subset:
            func = lambda x: self.data_sources.dataset.image_aspect_ratio(x)
        elif type(self.data_sources) == ConcatDataset:
            # print(len(self.data_sources.datasets[0]))
            # print(len(self.data_sources.datasets[1]))
            # print('NumClass0:', self.data_sources.datasets[0].num_classes())
            # print('NumClass1:', self.data_sources.datasets[1].num_classes())
            def func(x):
                first_len = len(self.data_sources.datasets[0])
                # print('x:', x, end=' ')
                if x <= first_len-1:
                    return self.data_sources.datasets[0].image_aspect_ratio(x)
                else:
                    return self.data_sources.datasets[1].image_aspect_ratio(x - first_len)
        else:
            func = lambda x: self.data_sources.image_aspect_ratio(x)

        order.sort(key=func)

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]


if __name__ == '__main__':
    from augmentation import get_augumentation
    
    dataset = CocoDataset(root_dir='d:\\show', set_name='train_small',
                          transform=get_augumentation(phase='train'), do_aug=1)
    sample = dataset[0]
    print('image.shape', sample['image'].shape)
    print('bboxes:', sample['bboxes'])
    print('category_id:', sample['category_id'])