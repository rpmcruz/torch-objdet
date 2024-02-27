import argparse
parser = argparse.ArgumentParser()
parser.add_argument('datasets', nargs='+')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--imgsize', type=int, default=512)
parser.add_argument('--fast', action='store_true')
args = parser.parse_args()

import torch
import torchvision
import torchmetrics
import albumentations as A
from albumentations.pytorch import ToTensorV2
from time import time
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

################################## DATA ##################################

transform = A.Compose([
    A.Resize(int(args.imgsize*1.1), int(args.imgsize*1.1)),
    A.RandomCrop(args.imgsize, args.imgsize),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(p=1),
    A.Normalize(0, 1),
    ToTensorV2(),
], A.BboxParams('pascal_voc'))

tr = getattr(data, args.dataset)('/data/auto', 'train', transform)
tr = torch.utils.data.DataLoader(tr, args.batchsize, True, num_workers=4, pin_memory=True, collate_fn=data.detection_collate)

transform = A.Compose([
    A.Resize(args.imgsize, args.imgsize),
    A.Normalize(0, 1),
    ToTensorV2(),
], A.BboxParams('pascal_voc'))

ts = getattr(data, args.dataset)('/data/auto', 'test', transform)
ts = torch.utils.data.DataLoader(ts, args.batchsize, num_workers=4, pin_memory=True, collate_fn=data.detection_collate)

################################## MODEL ##################################

# torchvision has the following object detection models:
# https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
model = torchvision.models.detection.fcos_resnet50_fpn(
    num_classes=data.num_classes, min_size=args.imgsize).to(device)

################################## OPTS ##################################

opts = torch.optim.Adam(model.parameters(), 1e-4)

################################## TRAIN ##################################

for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    model.train()
    for d in tr:
        images, targets = d
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        out_losses = model(images, targets)
        loss = sum(l for l in out_losses.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        avg_loss += float(loss) / len(ds)
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Loss: {avg_loss}')
    metric = torchmetrics.detection.mean_ap.MeanAveragePrecision().to(device)
    model.eval()
    for d in ds:
        images, targets = d
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        preds = model(images)
        metric.update(preds, targets)
    print(f'Evaluate - mAP: {metric.compute()["map"].item()}')

output = '-'.join(args.datasets)
torch.save(model.cpu(), f'model-{output}.pth')
