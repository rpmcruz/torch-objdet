import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

from time import time
import torch
from torchvision.transforms import v2
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################ DATA ############################

transform = v2.Compose([
    v2.Resize((1024, 1024)),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, True),
])

dataset = data.Kitti('/data/auto', transform)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, True, collate_fn=lambda x: x, num_workers=4, pin_memory=True)
test_dataloader = DataLoader(test_dataset, args.batch_size, collate_fn=lambda x: x, num_workers=4, pin_memory=True)

############################ LOOP ############################

model = torchvision.models.detection.fcos_resnet50_fpn(
    num_classes=data.Kitti.num_classes).to(device)
opt = torch.optim.Adam(model.parameters())

for epoch in range(args.epochs):
    tic = time()
    avg_loss = 0
    model.train()
    for imgs, targets in train_dataloader:
        imgs = imgs.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        sup_loss = model(imgs, targets)
        total_loss = sum(l for l in sup_loss.values())

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        avg_loss += float(total_loss) / len(train_dataloader)

    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Loss: {avg_loss}')

print(f'Saving model...')
torch.save(model.cpu(), args.save_model)
