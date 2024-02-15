import torch
import torchvision
from torch.nn import MSELoss
from train import MVTecDataset
import os
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['train','test'], default='train')
parser.add_argument('--dataset_path', default=r'../mvtec_anomaly_detection')
parser.add_argument('--category', default='bottle')
parser.add_argument('--num_epochs', default=10)
parser.add_argument('--batch_size', default=32)
parser.add_argument('--load_size', default=256)
parser.add_argument('--input_size', default=224)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])

gt_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size)])

image_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=data_transforms, gt_transform=gt_transforms, phase='train')
train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0)

model = torchvision.models.wide_resnet50_2(pretrained=True)

model.fc = torch.nn.Linear(in_features=2048, out_features=100)

criterion = MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# print(next(iter(train_loader))) 
for epoch in range(args.num_epochs):
    for data in tqdm(train_loader):
        print(data[-1])