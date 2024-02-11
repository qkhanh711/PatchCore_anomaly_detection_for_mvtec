# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 07月 20日 星期二 10:32:12 CST
@Description: path core 训练
'''

import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# from data import HxqData
from model import WideResnet502
from mvtec import MVTecDataset
from memory_bank import MemoryBank
"""

"""

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--img_dir",type=str,default="../mvtec_anomaly_detection")
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--compress_rate",type=float,default=0.01)
    parser.add_argument("--save_dir",type=str,default="memoryBankSaveDir")
    parser.add_argument("--input_size",type=tuple,default=416)
    parser.add_argument("--device",type=str,default="cuda")
    args=parser.parse_args()
    return args

def train(args):
    device=torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    dataset=MVTecDataset(args.img_dir,classname=None,resize=256,imagesize=args.input_size)
    dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    # print_shape
    print(next(iter(dataloader))['image'].shape)

    net=WideResnet502().to(device)
    net.eval()
    print("model init done")

    embedding_bank=[]
    print("start get features")
    with torch.no_grad():
        for data in tqdm(dataloader):
            data['image']=data['image'].to(device)
            z=net(data['image'])
            features=z.detach().cpu().numpy()
            features=np.transpose(features,(0,2,3,1))
            embedding_bank.append(features.reshape(-1,features.shape[-1]))
    print("get features done,generate memory bank")
    embedding_bank=np.concatenate(embedding_bank,axis=0)
    memory_bank_dealer=MemoryBank().to(device)
    memory_bank_dealer.bank_generate(embedding_bank,int(embedding_bank.shape[0]*args.compress_rate),args.save_dir)
    print("generate memory bank done")

if __name__=="__main__":
    args=parse_args()
    train(args)



