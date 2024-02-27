import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model')
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--imgsize', type=int, default=512)
args = parser.parse_args()

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchmetrics
from time import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

################################## DATA ##################################

transform = A.Compose([
    A.Resize(args.imgsize, args.imgsize),
    A.Normalize(0, 1),
    ToTensorV2(),
], A.BboxParams('pascal_voc'))

ts = ds = getattr(data, args.dataset)('test', transform)
ts = torch.utils.data.DataLoader(ts, args.batchsize, num_workers=4, pin_memory=True, collate_fn=data.detection_collate)

################################## MODEL ##################################

model = torch.load(args.model, map_location=device)

################################## EVAL ##################################

def draw_bboxes(boxes, labels):
    for box, k in zip(boxes, labels):
        box = box.cpu()
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], ec='cyan', lw=2, fill=False))
        label = ds.objects[k-1] if hasattr(ds, 'objects') else str(k)
        plt.text(box[0], box[1], label, c='cyan', va='bottom')

from torchmetrics.detection import mean_ap
metric = mean_ap.MeanAveragePrecision().to(device)

model.eval()
for i, d in enumerate(ts):
    tic = time()
    image, _, target = d
    image = image.to(device)
    target = [{k: v.to(device) for k, v in t.items()} for t in target]
    with torch.no_grad():
        out, mask, uncertainty = model(image, args.threshold)
    metric.update(out, target)

    toc = time()
    if args.visualize:
        matplotlib.rcParams['figure.figsize'] = (0.75*image.shape[3]/100, 0.75*image.shape[2]/100)
        ncolors = args.visualize_ncolors if args.visualize_ncolors else ds.num_classes
        cmap = 'gray_r' if ncolors == 2 else 'tab10'
        fout = f'visualize-{args.dataset}-{i:02d}-image.jpg'
        plt.clf(); plt.axis('off')
        plt.imshow(image[0].permute(1, 2, 0).cpu())
        draw_bboxes(target[0]['boxes'], target[0]['labels'])
        plt.savefig(fout, bbox_inches='tight', pad_inches=0)

        fout = f'visualize-{args.dataset}-{i:02d}-uncertainty.png'
        plt.clf(); plt.axis('off')
        plt.imshow(uncertainty[0, 0].cpu(), vmin=0, vmax=1, cmap='gray')
        plt.savefig(fout, bbox_inches='tight', pad_inches=0)

        fout = f'visualize-{args.dataset}-{i:02d}-mask-{args.threshold}.png'
        plt.clf(); plt.axis('off')
        plt.imshow(mask[0, 0].cpu(), vmin=0, vmax=1, cmap='gray')
        plt.savefig(fout, bbox_inches='tight', pad_inches=0)

        plt.clf(); plt.axis('off')
        fout = f'visualize-{args.dataset}-{i:02d}-pred-{args.threshold}.jpg'
        plt.imshow(image[0].permute(1, 2, 0).cpu())
        draw_bboxes(out[0]['boxes'], out[0]['labels'])
        plt.savefig(fout, bbox_inches='tight', pad_inches=0)

metric = metric.compute()
metric = metric['map'].item()
print(args.dataset, metric, sep=',')
