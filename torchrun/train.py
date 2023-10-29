import torch.distributed as dist
from sched import scheduler
from model import *
from dataset import *
import torch
import torch.nn as nn
from torch.nn import functional as F 
import torch.distributed as dist

rank = dist.get_rank()

def eval_and_clean_label(model,val_loader,args):

    model.eval()
    
    for _, (img_tensor,label,data_path) in enumerate(val_loader):
        bacth_size = img_tensor.shape[0]
        img_tensor = img_tensor.to(device)
        label = label.to(device)

        output = model(img_tensor)

        prob = F.sigmoid(output).view(-1)

        is_greater_threshold = prob > 0.8
        is_greater_threshold_index = is_greater_threshold.nonzero()[:,0]
        
        for index in range(is_greater_threshold_index.shape[0]):
            img_path = data_path[index]
            name2label[img_path] = 1
            if rank == 0:
                print(img_path)

    json.dump(name2label,open('name2label.json','w'))
    model.train()

def eval(model,test_loader,args):

    model.eval()
    all_eval_num, all_correct_eval_num = 0., 0.

    for _, (img_tensor,label,data_path) in enumerate(test_loader):
        bacth_size = img_tensor.shape[0]
        img_tensor = img_tensor.to(device)
        label = label.to(device)

        output = model(img_tensor)

        prob = F.sigmoid(output).view(-1)

        pred = torch.round(prob).long()
        correct_num = (pred == label).sum()

        batch_all_eval_num_list,bacth_all_correct_eval_num_list = [torch.zeros(1).to(device) for _ in range(args.world_size)],[torch.zeros(1).to(device) for _ in range(args.world_size)]
        all_batch_size = torch.tensor(bacth_size).to(device)
        dist.all_gather(batch_all_eval_num_list, all_batch_size)
        dist.barrier()
        dist.all_reduce(all_batch_size, op=dist.ReduceOp.SUM)
        dist.barrier()

        dist.all_gather(bacth_all_correct_eval_num_list, correct_num)
        dist.barrier()
        dist.all_reduce(correct_num, op=dist.ReduceOp.SUM)
        dist.barrier()

        all_eval_num += all_batch_size.item()
        all_correct_eval_num += correct_num.item()

        if rank == 0:
            print("%d/%d"%(all_correct_eval_num,all_eval_num))

    if rank == 0:

        print("all eval num: %f,all correct eval num: %f, acc: %f" %(all_eval_num, all_correct_eval_num, all_correct_eval_num/all_eval_num))
        
        print("%d/%d"%(correct_num.item(),bacth_size))
    model.train()


def reinit_dataloader():
    train_dataset = GoodsDataset(data_dir = all_data_dir,names=train_img_groups) 
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,batch_size=12,sampler=train_sampler)
    return train_loader

import argparse
from utils import get_optimizer

parser = argparse.ArgumentParser(
                    prog='sole clf',
                    description='tudo',
                    )
parser.add_argument('--resume', action='store_true', default=False, help='Resume training')
parser.add_argument('--epochs', type=int, default=30, help='number of total epochs to run')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer type') 
parser.add_argument("--lr",type=float,default=0.0005)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--alpha', default=0.99, type=float, metavar='M',
                         help='alpha for ')
parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
parser.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_fc_times', '--lft', default=5, type=int,
                    metavar='LR', help='initial model last layer rate')
parser.add_argument('--begin_eval_epoch',default=1, type=int, help='frequency of eval')
parser.add_argument('--world_size',default=4,type=int)
parser.add_argument('--checkpoint_path',default='./model.checkpoint',type=str,help='resume model checkpoint path')
args = parser.parse_args()

if args.resume:
    print('\033[31m' + 'Train from resume checkpoint...' + '\033[0m')
    print('\033[31m' + 'checkoint model path is {}'.format(args.checkpoint_path) + '\033[0m')
    map_location = {"cuda:0":f"cuda:{local_rank}"}
    model.load_state_dict(torch.load(args.checkpoint_path,map_location=map_location))
    train_loader = reinit_dataloader()
    eval(model, test_loader, args)
else:
    if rank == 0:
        torch.save(model.state_dict(), args.checkpoint_path)
    dist.barrier()

optimizer = get_optimizer(model, args)
model.train()

for epoch in  range(args.epochs):
    train_sampler.set_epoch(epoch)

    
    for i, (img_tensor,label,data_path) in enumerate(train_loader):

        optimizer.zero_grad()
        img_tensor = img_tensor.to(device)
        label = label.to(device).view(-1,1).to(img_tensor.dtype)

        output = model(img_tensor)

        loss = criterion(output,label)
        loss.backward()
        optimizer.step()

        if rank == 0 and i % 5 == 0:
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
    if epoch >= args.begin_eval_epoch and epoch % 4 == 0:
        eval_and_clean_label(model,train_loader,args)
        train_loader = reinit_dataloader()
    if rank == 0:
        eval(model, test_loader, args)
        torch.save(model.state_dict(), args.checkpoint_path)
    dist.barrier()
    map_location = {"cuda:0":f"cuda:{local_rank}"}
    model.load_state_dict(torch.load(args.checkpoint_path,map_location=map_location))
    



