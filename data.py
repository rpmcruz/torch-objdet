import torchvision, torch
import torchvision.tv_tensors
import numpy as np
import os

class Kitti:
    NUM_CLASSES = 3
    class_map = {'Car': 0, 'Van': 0, 'Cyclist': 1, 'Pedestrian': 2, 'Person_sitting': 2}

    def __init__(self, root, transform=None):
        self.root = os.path.join(root, 'kitti', 'object', 'training')
        self.files = sorted(os.listdir(os.path.join(self.root, 'image_2')))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_fname = os.path.join(self.root, 'image_2', self.files[i])
        gt_fname = os.path.join(self.root, 'label_2', self.files[i][:-3] + 'txt')
        image = torchvision.tv_tensors.Image(torchvision.io.decode_image(img_fname))
        labels = np.loadtxt(gt_fname, str, usecols=[0], ndmin=1)
        ix = [label in self.class_map for label in labels]
        targets = {
            'labels': torch.tensor([self.class_map[label] for label in labels[ix]]),
            'boxes': torchvision.tv_tensors.BoundingBoxes(np.loadtxt(gt_fname, np.float32, usecols=range(4, 8), ndmin=2)[ix], format=torchvision.tv_tensors.BoundingBoxFormat.XYXY, canvas_size=image.shape[1:])
        }
        if self.transform:
            image, targets = self.transform(image, targets)
        return image, targets
