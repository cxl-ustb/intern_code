from torchvision.models.resnet import resnet50
import torch
import torch.nn as nn
import os
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.distributed as dist
dist.init_process_group(backend='nccl',init_method='env://')
model = resnet50(pretrained=True)

model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)

local_rank = int(os.environ['LOCAL_RANK'])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

model = model.cuda()
model = DistributedDataParallel(model, device_ids=[local_rank])

criterion = nn.BCEWithLogitsLoss().to(device)