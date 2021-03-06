# import inspect

# __print__ = print

# def print(*args):
#     callerframerecord = inspect.stack()[1]
#     frame = callerframerecord[0]
#     info = inspect.getframeinfo(frame)
#     __print__(args, info.filename, info.function, info.lineno)


import os
import platform

if platform.system() == 'Linux':
    import subprocess
    import sys
    import shutil

    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    install('install/pycocotools-2.0-cp36-cp36m-linux_x86_64.whl')
    # install('install/pytoan-0.6.4-py3-none-any.whl')
    # install('install/imgaug-0.2.6-py3-none-any.whl')
    # install('install/albumentations-0.4.5-py3-none-any.whl')
    

import argparse
import collections
import platform
import numpy as np

import torch
import torch.optim as optim
import torchvision
from torchvision import transforms

from retinanet import model
from retinanet.augmentation import get_augumentation
from retinanet.dataloader import H5CoCoDataset, CocoDataset, detection_collate, collater, Resizer, \
    AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader, ConcatDataset, Subset

from retinanet import coco_eval
import json
import pandas as pd
from io import StringIO

assert torch.__version__.split('.')[0] == '1'


########################################
# DEPTH = 101250  # 使用resnet101模型,但載入resnet50權重
# IMAGE_SIZE = (540, 960)
# BATCH_SIZE = 5

# DEPTH = 101250  # 使用resnet101模型,但載入resnet50權重
# IMAGE_SIZE = (675, 1200)
# BATCH_SIZE = 4

DEPTH = 50
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 10

# DEPTH = 50
# IMAGE_SIZE = (540, 960)
# BATCH_SIZE = 8

# DEPTH = 50
# IMAGE_SIZE = (675, 1200)
# BATCH_SIZE = 6

##########

EPOCHS = 40
NUM_WORKERS = 2
PRETRAINED = True
MERGE_VAL = 0
DO_AUG = 0

# LR_CHOICE = 'lr_scheduler'
LR = 1e-4
PATIENCE = 3
FACTOR = 0.1

# LR_CHOICE = 'lr_map'
LR_MAP = {'1':'2e-4', '25':'1.5e-4', '30':'7.5e-5', '35':'3e-5'}

LR_CHOICE = 'lr_fn'
LR_FN = {
    'LR_START': '1e-5',
    'LR_MAX': '1e-4',
    'LR_MIN': '1e-5',
    'LR_RAMPUP_EPOCHS': '10',
    'LR_SUSTAIN_EPOCHS': '5',
    'LR_EXP_DECAY': '.8'
}
########################################


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_change_map(epoch, lr, lr_map):
    new_lr = lr
    for k in lr_map.keys():
        if epoch >= int(k):
            new_lr = float(lr_map[k])

    if new_lr != lr:
        print('changing lr form {} to {}'.format(lr, new_lr))
    
    return new_lr


def lrfn(epoch, lr_fn_dicts):
    if epoch < int(lr_fn_dicts['LR_RAMPUP_EPOCHS']):
        lr = (float(lr_fn_dicts['LR_MAX']) - float(lr_fn_dicts['LR_START'])) / int(lr_fn_dicts['LR_RAMPUP_EPOCHS']) * epoch + \
            float(lr_fn_dicts['LR_START'])
    elif epoch < int(lr_fn_dicts['LR_RAMPUP_EPOCHS']) + int(lr_fn_dicts['LR_SUSTAIN_EPOCHS']):
        lr = float(lr_fn_dicts['LR_MAX'])
    else:
        lr = (float(lr_fn_dicts['LR_MAX']) - float(lr_fn_dicts['LR_MIN'])) * \
            float(lr_fn_dicts['LR_EXP_DECAY'])**(epoch - int(lr_fn_dicts['LR_RAMPUP_EPOCHS']) - \
            int(lr_fn_dicts['LR_SUSTAIN_EPOCHS'])) + float(lr_fn_dicts['LR_MIN'])
    return lr


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='show')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='/mnt/marathon')
    parser.add_argument('--image_size', help='image size', type=int, nargs=2, default=IMAGE_SIZE)
    parser.add_argument('--limit', help='limit', type=int, nargs=2, default=(0, 0))
    parser.add_argument('--batch_size', help='batch size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_works', help='num works', type=int, default=NUM_WORKERS)
    parser.add_argument('--num_classes', help='num classes', type=int, default=3)
    parser.add_argument('--merge_val', help='merge_val', type=int, default=MERGE_VAL)
    parser.add_argument('--do_aug', help='do_aug', type=int, default=DO_AUG)
    parser.add_argument('--lr_choice', default=LR_CHOICE, choices=['lr_scheduler', 'lr_map', 'lr_fn'], type=str)
    parser.add_argument('--lr', help='lr', type=float, default=LR)
    parser.add_argument("--lr_map", dest="lr_map", action=StoreDictKeyPair, default=LR_MAP)
    parser.add_argument("--lr_fn", dest="lr_fn", action=StoreDictKeyPair, default=LR_FN)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=DEPTH)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=EPOCHS)
    parser = parser.parse_args(args)

    print('dataset:', parser.dataset)
    print('depth:', parser.depth)
    print('epochs:', parser.epochs)
    print('image_size:', parser.image_size)
    print('batch_size:', parser.batch_size)
    print('num_works:', parser.num_works)
    print('merge_val:', parser.merge_val)
    print('do_aug:', parser.do_aug)
    print('lr_choice:', parser.lr_choice)
    print('lr:', parser.lr)
    print('lr_map:', parser.lr_map)
    print('lr_fn:', parser.lr_fn)
    print('num_classes:', parser.num_classes)
    print('limit:', parser.limit)

    # Create the data loaders
    # dataset_train, _ = torch.utils.data.random_split(dataset_train, [NUM_COCO_DATASET_TRAIN, len(dataset_train) - NUM_COCO_DATASET_TRAIN])
    # dataset_val, _ = torch.utils.data.random_split(dataset_val, [NUM_COCO_DATASET_VAL, len(dataset_val) - NUM_COCO_DATASET_VAL])

    transform_train = None
    transform_vail = None
    collate_fn = None
    if parser.do_aug:
        transform_train = get_augumentation('train', parser.image_size[0], parser.image_size[1])
        transform_vail = get_augumentation('test', parser.image_size[0], parser.image_size[1])
        collate_fn = detection_collate
    else:
        transform_train = transforms.Compose([
            # Normalizer(),
            # Augmenter(),
            Resizer(*parser.image_size)])
        transform_vail = transforms.Compose([
            # Normalizer(), 
            Resizer(*parser.image_size)])
        collate_fn = collater

    if parser.dataset == 'h5':
        dataset_train = H5CoCoDataset('{}/train_small.hdf5'.format(parser.coco_path), 'train_small')
        dataset_val = H5CoCoDataset('{}/test.hdf5'.format(parser.coco_path), 'test')
    else:
        dataset_train = CocoDataset(parser.coco_path, set_name='train_small', do_aug=parser.do_aug,
            transform=transform_train, limit_len=parser.limit[0])
        dataset_val = CocoDataset(parser.coco_path, set_name='test', do_aug=parser.do_aug,
            transform=transform_vail, limit_len=parser.limit[1])

    # 混合val
    if parser.merge_val:
        dataset_train += dataset_val

    print('training images: {}'.format(len(dataset_train)))
    print('val images: {}'.format(len(dataset_val)))
    
    steps_pre_epoch = len(dataset_train) // parser.batch_size
    print('steps_pre_epoch:', steps_pre_epoch)

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, batch_size=1, num_workers=parser.num_works, shuffle=False,
        collate_fn=collate_fn, batch_sampler=sampler)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=parser.num_classes, pretrained=PRETRAINED)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=parser.num_classes, pretrained=PRETRAINED)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=parser.num_classes, pretrained=PRETRAINED)
    elif parser.depth == 101250:
        retinanet = model.resnet101with50weight(num_classes=parser.num_classes, pretrained=PRETRAINED)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=parser.num_classes, pretrained=PRETRAINED)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=parser.num_classes, pretrained=PRETRAINED)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()
    retinanet.training = True

    if parser.lr_choice == 'lr_map':
        lr_now = lr_change_map(1, 0, parser.lr_map)
    elif parser.lr_choice == 'lr_fn':
        lr_now = float(parser.lr_fn['LR_START'])
    elif parser.lr_choice == 'lr_scheduler':
        lr_now = parser.lr

    # optimizer = optim.Adam(retinanet.parameters(), lr=lr_now)
    optimizer = optim.AdamW(retinanet.parameters(), lr=lr_now)
    # optimizer = optim.SGD(retinanet.parameters(), lr=lr_now, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.SGD(retinanet.parameters(), lr=lr_now)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE, factor=FACTOR, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    iteration_loss_path = 'iteration_loss.csv'
    if os.path.isfile(iteration_loss_path):
        os.remove(iteration_loss_path)
    
    epoch_loss_path = 'epoch_loss.csv'
    if os.path.isfile(epoch_loss_path):
        os.remove(epoch_loss_path)
    
    eval_train_path = 'eval_train_result.csv'
    if os.path.isfile(eval_train_path):
        os.remove(eval_train_path)

    eval_val_path = 'eval_val_result.csv'
    if os.path.isfile(eval_val_path):
        os.remove(eval_val_path)

    USE_KAGGLE = True if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', False) else False
    if USE_KAGGLE:
        iteration_loss_path = '/kaggle/working/' + iteration_loss_path
        epoch_loss_path = '/kaggle/working/' + epoch_loss_path
        eval_val_path = '/kaggle/working/' + eval_val_path
        eval_train_path = '/kaggle/working/' + eval_train_path

    with open(epoch_loss_path, 'a+') as epoch_loss_file, \
         open(iteration_loss_path, 'a+') as iteration_loss_file, \
         open(eval_train_path, 'a+') as eval_train_file, \
         open(eval_val_path, 'a+') as eval_val_file:

        epoch_loss_file.write('epoch_num,mean_epoch_loss\n')
        iteration_loss_file.write('epoch_num,iteration,classification_loss,regression_loss,iteration_loss\n')
        eval_train_file.write('epoch_num,map50\n')
        eval_val_file.write('epoch_num,map50\n')

        for epoch_num in range(parser.epochs):
            retinanet.train()
            retinanet.module.freeze_bn()

            epoch_loss = []
            for iter_num, data in enumerate(dataloader_train):
                optimizer.zero_grad()
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                iteration_loss = np.mean(loss_hist)
                print('\rEpoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                      epoch_num+1, iter_num+1, float(classification_loss), float(regression_loss), iteration_loss), end=' ' * 50)

                iteration_loss_file.write('{},{},{:1.5f},{:1.5f},{:1.5f}\n'.format(epoch_num+1,
                    epoch_num * steps_pre_epoch + (iter_num+1), float(classification_loss), float(regression_loss),
                    iteration_loss))
                iteration_loss_file.flush()

                del classification_loss
                del regression_loss

            mean_epoch_loss = np.mean(epoch_loss)
            epoch_loss_file.write('{},{:1.5f}\n'.format(epoch_num+1, mean_epoch_loss))
            epoch_loss_file.flush()

            if parser.lr_choice == 'lr_map':
                lr_now = lr_change_map(epoch_num+1, lr_now, parser.lr_map)
                adjust_learning_rate(optimizer, lr_now)
            elif parser.lr_choice == 'lr_fn':
                lr_now = lrfn(epoch_num+1, parser.lr_fn)
                adjust_learning_rate(optimizer, lr_now)
            elif parser.lr_choice == 'lr_scheduler':
                scheduler.step(mean_epoch_loss)

            # if parser.dataset != 'show':
            #     print('Evaluating dataset_train')
            #     coco_eval.evaluate_coco(dataset_train, retinanet, parser.dataset, parser.do_aug, eval_train_file, epoch_num)

            print('Evaluating dataset_val')
            coco_eval.evaluate_coco(dataset_val, retinanet, parser.dataset, parser.do_aug, eval_val_file, epoch_num)
    return parser


def write_result_csv(parser):
    test_file = '{}/test/test.json'.format(parser.coco_path)
    with open(test_file) as f:
        data = json.load(f)
        image_dict = {}
        for image in data['images']:
            image_dict[image['id']] = image['file_name']

    # 將test_bbox_results.json寫入result.csv
    RESULT_JSON = 'test_bbox_results.json'
    RESULT_CSV = 'result.csv'
    USE_KAGGLE = True if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', False) else False
    if USE_KAGGLE:
        RESULT_JSON = '/kaggle/working/' + RESULT_JSON
        RESULT_CSV  = '/kaggle/working/' + RESULT_CSV
    with open(RESULT_JSON) as f_result, open(RESULT_CSV, 'w') as f_out:
        anno_list = json.load(f_result)
        f_out.write("image_filename,label_id,x,y,w,h,confidence\n")
        for anno in anno_list:
            f_out.write("%s" % image_dict[anno['image_id']])
            x,y,w,h = anno['bbox']
            f_out.write(",{},{},{},{},{},{:.02f}".format(anno['category_id'], x,y,w,h, anno['score']))
            f_out.write('\n')


if __name__ == '__main__':
    print('CUDA available: {}'.format(torch.cuda.is_available()))
    parser = main()
    write_result_csv(parser)