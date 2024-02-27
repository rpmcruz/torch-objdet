import torch
import torchvision
import os
import json
import numpy as np
from skimage.io import imread

def detection_collate(batch):
    images = torch.stack([b['image'] for b in batch])
    targets = [{
        'boxes': torch.tensor([o[:4] for o in b['bboxes']], dtype=torch.float32) if len(b['bboxes']) else torch.empty(0, 4),
        'labels': torch.tensor([o[-1] for o in b['bboxes']], dtype=torch.int64) if len(b['bboxes']) else torch.empty(0, dtype=torch.int64),
    } for b in batch]
    return images, targets

num_classes = 8  # including background
labels = ['person', 'bike', 'motorbike', 'car', 'bus', 'truck', 'train']

class KITTI(torch.utils.data.Dataset):
    # ignoring Misc and DontCare
    # merging vans with cars because vans (in kitti) are considered cars (in bdd100k)
    objects = [['Pedestrian', 'Person_sitting'], ['Cyclist'], [], ['Car', 'Van'], [], ['Truck'], ['Tram']]

    def label_to_index(self, label):
        for i, objs in enumerate(self.objects):
            if label in objs:
                return i+1
        return None

    def __init__(self, root, fold, transforms=None, seed=123):
        self.transforms = transforms
        self.imgs_dir = os.path.join(root, 'kitti', 'object', 'training', 'image_2')
        labels_dir = os.path.join(root, 'kitti', 'object', 'training', 'label_2')
        self.images = sorted(os.listdir(self.imgs_dir))
        rand = np.random.RandomState(seed)
        ix = rand.choice(len(self.images), len(self.images), False)
        ix = ix[:int(len(ix)*0.8)] if fold == 'train' else ix[int(len(ix)*0.8):]
        self.images = [self.images[i] for i in ix]
        self.bboxes = []
        for fname in self.images:
            fname = os.path.join(labels_dir, fname[:-3] + 'txt')
            bboxes = np.loadtxt(fname, usecols=range(4, 8), ndmin=2)
            labels = np.loadtxt(fname, str, usecols=0, ndmin=1)
            bboxes = [list(bbox) + [self.label_to_index(label)]
                for bbox, label in zip(bboxes, labels) if self.label_to_index(label) != None]
            self.bboxes.append(bboxes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = imread(os.path.join(self.imgs_dir, self.images[i]))
        bboxes = self.bboxes[i]
        d = {'image': image, 'bboxes': bboxes}
        if self.transforms is not None:
            d = self.transforms(**d)
        return d

class BDD100K(torch.utils.data.Dataset):
    objects = ['person', 'bike', 'motor', 'car', 'bus', 'truck', 'train']

    def intersects(self, bbox1, bbox2):
        return not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])
    def process_bboxes(self, bboxes):
        # merge bounding box of rider with bike and rider with motor
        ret = []
        for bbox1 in bboxes:
            if bbox1[-1] not in ('rider', 'bike', 'motor'):
                ret.append(bbox1)
            elif bbox1[-1] in ('bike', 'motor'):
                intersects_with = None
                for bbox2 in bboxes:
                    if bbox2[-1] == 'rider':
                        if self.intersects(bbox1, bbox2):
                            intersects_with = bbox2
                            break
                if intersects_with != None:
                    bbox1 = (
                        min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),
                        max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3]),
                        bbox1[-1])
                ret.append(bbox1)
        return ret

    def __init__(self, root, fold, transforms=None):
        fold = 'val' if fold == 'test' else fold
        self.transforms = transforms
        self.imgs_dir = os.path.join(root, 'bdd100k', 'images', '100k', fold)
        labels_fname = os.path.join(root, 'bdd100k', 'labels', f'bdd100k_labels_images_{fold}.json')
        labels = json.load(open(labels_fname))
        self.images = sorted(os.listdir(self.imgs_dir))
        self.bboxes = {label['name']: self.process_bboxes([(obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], obj['box2d']['y2'], obj['category']) for obj in label['labels'] if 'box2d' in obj]) for label in labels}
        self.bboxes = {key: [bbox[:-1] + (self.objects.index(bbox[-1])+1,) for bbox in bboxes if bbox[-1] in self.objects] for key, bboxes in self.bboxes.items()}
        # there are some images for which we have no information: filter them
        self.images = [image for image in self.images if image in self.bboxes]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = imread(os.path.join(self.imgs_dir, self.images[i]))
        bboxes = self.bboxes[self.images[i]]
        d = {'image': image, 'bboxes': bboxes}
        if self.transforms is not None:
            d = self.transforms(**d)
        return d

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--only', nargs='+')
    args = parser.parse_args()
    ds = globals()[args.dataset]('/data/auto', 'train', None)
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    for i, d in enumerate(ds):
        if args.only and not any(bbox[-1]-1 in [ds.objects.index(t) for t in args.only] for bbox in d['bboxes']):
            continue
        plt.clf()
        plt.imshow(d['image'])
        for x1, y1, x2, y2, label in d['bboxes']:
            plt.gca().add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none'))
            plt.text(x1, y1, ds.objects[label-1], color='r')
        plt.title(f'{args.dataset} {i}')
        plt.show()
