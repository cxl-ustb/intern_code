
import torch.distributed as dist
from torchvision.models.resnet import resnet50
import torch.nn as nn
import os
import torch
from torch.utils.data.distributed import  DistributedSampler
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import Dataset,DataLoader
from torch.nn import functional as F

world_size = 1
def init_dist():
    dist.init_process_group(backend='nccl',init_method='env://')

# model
def init_model():
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[local_rank])
    checkpoint_path ='./model.checkpoint'
    map_location = {"cuda:0":f"cuda:{local_rank}"}
    model.load_state_dict(torch.load(checkpoint_path,map_location=map_location))

    return model,device

# infer dataset
def init_data(data_dir):
    transform = Compose(
        [   Resize(size=(800,800)),
            ToTensor(),
            Normalize((-0.1307 / 0.3081,), (1 / 0.3081,)),
        ]
    )
    
    class GoodsDataset(Dataset):
        def __init__(self,data_dir):
            super(GoodsDataset,self).__init__()

            self.data_dir = data_dir
            self.data_paths = sorted(os.listdir(self.data_dir))

        def __len__(self,):
            return len(self.data_paths)

        def __getitem__(self, index):
            data_path = self.data_paths[index]
            full_path = os.path.join(self.data_dir,data_path)
            
            img_rgb = Image.open(full_path)
            img_tensor = transform(img_rgb)

            return img_tensor,data_path

    val_dataset = GoodsDataset(data_dir=data_dir) 
    infer_sampler = DistributedSampler(val_dataset,shuffle=False)
    infer_loader = DataLoader(val_dataset,batch_size=8,sampler=infer_sampler)

    return infer_loader

def infer(model,data_loader,device):
    model.eval()

    
    for _, (img_tensor,data_path) in enumerate(data_loader):
        
        img_tensor = img_tensor.to(device)

        output = model(img_tensor)

        prob = F.sigmoid(output).view(-1)
        pred = torch.round(prob).long()


        print(pred)
        print(data_path)
        # this_correct_eval_num = torch.sum(max_index == label)
        # batch_all_eval_num_list,bacth_all_correct_eval_num_list = [torch.zeros(1).to(device) for _ in range(world_size)],[torch.zeros(1).to(device) for _ in range(world_size)]

        # all_batch_size = torch.tensor(bacth_size).to(device)
        # dist.all_gather(batch_all_eval_num_list, all_batch_size)
        # dist.barrier()
        # dist.all_reduce(all_batch_size, op=dist.ReduceOp.SUM)
        # dist.barrier()

        # dist.all_gather(bacth_all_correct_eval_num_list, this_correct_eval_num)
        # dist.barrier()
        # dist.all_reduce(this_correct_eval_num, op=dist.ReduceOp.SUM)
        # dist.barrier()

        # all_eval_num += all_batch_size.item()
        # all_correct_eval_num += this_correct_eval_num.item()



if __name__ == "__main__":
    data_dir = '/mnt/bn/image-bank-project/cxl/data'
    init_dist()
    model,device = init_model()
    data = init_data(data_dir)
    infer(model,data,device)


    



