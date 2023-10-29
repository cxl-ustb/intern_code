
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor,Compose,Normalize,Resize
import os
from PIL import Image
import torch
from torch.utils.data.distributed import  DistributedSampler
import json
import random

name2label = json.load(open('./name2label.json','r'))

transform = Compose(
    [   Resize(size=(800,800)),
        ToTensor(),
        Normalize((-0.1307 / 0.3081,), (1 / 0.3081,)),
    ]
)


class GoodsDataset(Dataset):
    def __init__(self,data_dir,names):
        super(GoodsDataset,self).__init__()

        self.data_dir = data_dir
        self.data_paths = names

    def __len__(self,):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.data_paths[index]

        full_path = os.path.join(self.data_dir,data_path)
        if data_path not in name2label:
            print(data_path)
        label = name2label[data_path]
        label = torch.tensor(label)
        
        img_rgb = Image.open(full_path)
        img_tensor = transform(img_rgb)

        return img_tensor,label,data_path

all_data_dir = '/mnt/bn/image-bank-project/cxl/data'
imgs = sorted(os.listdir(all_data_dir))
img_nums = len(imgs)
start,end = 0,img_nums-1
all_indexs = [i for i in range(img_nums)]
num_numbers = 0.2*img_nums
numbers = set()  # 使用set数据结构来存储不同的数

while len(numbers) < num_numbers:
    number = random.randint(start, end)
    numbers.add(number)\


number = list(numbers)
test_img_indexs = number
train_img_indexs = list(set(all_indexs) - numbers)

train_img_groups = [imgs[i] for i in train_img_indexs]
test_img_groups = [imgs[i] for i in test_img_indexs]

train_dataset = GoodsDataset(data_dir = all_data_dir,names=train_img_groups)
test_dataset = GoodsDataset(data_dir = all_data_dir,names=test_img_groups) 

train_sampler = DistributedSampler(train_dataset,shuffle=True)
test_sampler = DistributedSampler(test_dataset)

train_loader = DataLoader(train_dataset,batch_size=8,sampler=train_sampler)
test_loader = DataLoader(test_dataset,batch_size=8,sampler=test_sampler)  
print(len(train_dataset))
print(len(test_dataset))