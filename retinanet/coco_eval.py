import os
from pycocotools.cocoeval import COCOeval
import json
import torch
from torch.utils.data import Subset

def evaluate_coco(dataset, model, _type, threshold=0.05):
    model.eval()

    with torch.no_grad():
        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    if type(dataset) == Subset:
                        image_id = dataset.dataset.image_ids[index]
                        category_id = dataset.dataset.label_to_coco_label(label)
                    else:
                        image_id = dataset.image_ids[index]
                        category_id = dataset.label_to_coco_label(label)
                    image_result = {
                        'image_id'    : image_id,
                        'category_id' : category_id,
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            if type(dataset) == Subset:
                image_id = dataset.dataset.image_ids[index]
            else:
                image_id = dataset.image_ids[index]
            image_ids.append(image_id)

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        if type(dataset) == Subset:
            name = dataset.dataset.set_name
        else:
            name = dataset.set_name

        JSON_PATH = '{}_bbox_results.json'.format(name)
        USE_KAGGLE = True if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', False) else False
        if USE_KAGGLE:
            JSON_PATH = '/kaggle/working/' + JSON_PATH
        json.dump(results, open(JSON_PATH, 'w'), indent=4)

        if _type != '':
            # load results in COCO evaluation tool
            coco_true = dataset.coco
            coco_pred = coco_true.loadRes(JSON_PATH)

            # run COCO evaluation
            coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
            coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            model.train()
        return