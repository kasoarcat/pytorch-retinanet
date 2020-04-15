import os
import platform

if platform.system() == 'Linux':
    import subprocess
    import sys
    import shutil

    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    install('install/pycocotools-2.0.0-cp36-cp36m-linux_x86_64.whl')
    
import argparse
import collections
import platform
import numpy as np

import torch
import torch.optim as optim
import torchvision
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import H5CoCoDataset, CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader, ConcatDataset, Subset

from retinanet import coco_eval
import json
import pandas as pd
from io import StringIO

assert torch.__version__.split('.')[0] == '1'

##########
# DEPTH = 101250  # 使用resnet101模型,但載入resnet50權重
DEPTH = 50
EPOCHS = 50
PRETRAINED = True
BATCH_SIZE = 8
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
IMAGE_SIZE = (540, 960)
PATIENCE = 3
FACTOR = 0.1
##########

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='/mnt/marathon')
    parser.add_argument('--image_size', help='image size', type=int, nargs=2, default=IMAGE_SIZE)
    parser.add_argument('--batch_size', help='batch size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_works', help='num works', type=int, default=NUM_WORKERS)
    parser.add_argument('--lr', help='lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=DEPTH)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=EPOCHS)
    parser = parser.parse_args(args)

    print('dataset:', parser.dataset)
    print('depth:', parser.depth)
    print('epochs:', parser.epochs)
    print('image_size:', parser.image_size)
    print('batch_size:', parser.batch_size)
    print('num_works:', parser.num_works)
    print('lr:', parser.lr)

    # Create the data loaders
    if parser.dataset == 'limit':
        print('using limit dataset')
        dataset_train = CocoDataset(parser.coco_path, set_name='train_small',
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer(*parser.image_size)]),
                                # transform=get_augumentation('train', parser.image_size[0], parser.image_size[1]),
                                limit_len=2000)
        dataset_val = CocoDataset(parser.coco_path, set_name='test',
                              transform=transforms.Compose([Normalizer(), Resizer(*parser.image_size)]),
                              limit_len=200)
        # dataset_train, _ = torch.utils.data.random_split(dataset_train, [NUM_COCO_DATASET_TRAIN, len(dataset_train) - NUM_COCO_DATASET_TRAIN])
        # dataset_val, _ = torch.utils.data.random_split(dataset_val, [NUM_COCO_DATASET_VAL, len(dataset_val) - NUM_COCO_DATASET_VAL])
    elif parser.dataset == 'h5':
        print('using h5 dataset')
        dataset_train = H5CoCoDataset('{}/train_small.hdf5'.format(parser.coco_path), 'train_small')
        dataset_val = H5CoCoDataset('{}/test.hdf5'.format(parser.coco_path), 'test')
    else:
        print('using all dataset')
        dataset_train = CocoDataset(parser.coco_path, set_name='train_small',
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer(*parser.image_size)])
                                # transform=get_augumentation('train', parser.image_size[0], parser.image_size[1])
                                )
        dataset_val = CocoDataset(parser.coco_path, set_name='test',
                              transform=transforms.Compose([Normalizer(), Resizer(*parser.image_size)]))

    # 混合test
    dataset_train += dataset_val

    print('training images: {}'.format(len(dataset_train)))
    print('val images: {}'.format(len(dataset_val)))
    
    steps_pre_epoch = len(dataset_train) // parser.batch_size
    print('steps_pre_epoch:', steps_pre_epoch)

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, batch_size=1, num_workers=parser.num_works, shuffle=False, collate_fn=collater,
        batch_sampler=sampler)

    # Create the model
    if parser.dataset == '':
        num_classes = 3
    else:
        num_classes = 80
    print('num_classes:', num_classes)

    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=num_classes, pretrained=PRETRAINED)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=num_classes, pretrained=PRETRAINED)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=num_classes, pretrained=PRETRAINED)
    elif parser.depth == 101250:
        retinanet = model.resnet101with50weight(num_classes=num_classes, pretrained=PRETRAINED)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=num_classes, pretrained=PRETRAINED)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=num_classes, pretrained=PRETRAINED)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()
    retinanet.training = True

    # optimizer = optim.Adam(retinanet.parameters(), lr=LEARNING_RATE)
    optimizer = optim.AdamW(retinanet.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(retinanet.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.SGD(retinanet.parameters(), lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE, factor=FACTOR, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    iteration_loss_path = "iteration_loss.csv"
    if os.path.isfile(iteration_loss_path):
        os.remove(iteration_loss_path)
    
    epoch_loss_path = "epoch_loss.csv"
    if os.path.isfile(epoch_loss_path):
        os.remove(epoch_loss_path)
    
    USE_KAGGLE = True if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', False) else False
    if USE_KAGGLE:
        iteration_loss_path = '/kaggle/working/' + iteration_loss_path
        epoch_loss_path = '/kaggle/working/' + epoch_loss_path

    with open (epoch_loss_path, 'a+') as epoch_loss_file, open (iteration_loss_path, 'a+') as iteration_loss_file:
        epoch_loss_file.write('epoch_num,mean_epoch_loss\n')
        iteration_loss_file.write('epoch_num,iteration,classification_loss,regression_loss,iteration_loss\n')
        for epoch_num in range(parser.epochs):
            retinanet.train()
            retinanet.module.freeze_bn()

            epoch_loss = []
            for iter_num, data in enumerate(dataloader_train):
                try:
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
                except Exception as e:
                    print(e)
                    continue

            mean_epoch_loss = np.mean(epoch_loss)
            scheduler.step(mean_epoch_loss)
            epoch_loss_file.write('{},{:1.5f}\n'.format(epoch_num+1, mean_epoch_loss))
            epoch_loss_file.flush()

            print('Evaluating dataset')
            coco_eval.evaluate_coco(dataset_val, retinanet, parser.dataset)
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