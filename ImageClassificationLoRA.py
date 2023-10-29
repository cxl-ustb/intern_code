import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load()
from datasets import load_dataset
from transformers import AutoImageProcessor
import os
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoModelForImageClassification, AdamW, get_scheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import argparse
import torch.distributed as dist
from accelerate import Accelerator
accelerator = Accelerator(project_dir='.')

result_dir = './checkpoint'
os.makedirs(result_dir, exist_ok=True)
rank = torch.distributed.get_rank()
parser = argparse.ArgumentParser(
    prog = "lora image clf"
)
parser.add_argument("--world_size", default=8, type=int, help="World size")
args = parser.parse_args()

def train_lora(model, num_epochs, train_dataloader, val_dataloader , optimizer, lr_scheduler, progress_bar):
    model.train()
    prev_acc = 0.

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            if rank == 0:
                print(loss.item())
        acc = val_lora(model, val_dataloader)

        if rank == 0:
            if acc > prev_acc:
                prev_acc = acc
                print("save state to ...")
                checkpoint = accelerator.get_state_dict(model)
                torch.save(checkpoint, f"{result_dir}/model_best.pt")

def val_lora(model, val_dataloader):
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    
    all_eval_num, all_correct_eval_num = 0., 0.
    for batch in val_dataloader:
        outputs = model(**batch)
        logits = outputs["logits"].view(1,-1)
        labels = batch["labels"].view(1,-1).to(torch.int32)
        for (prob, label) in zip(logits, labels):
            batch_size = label.shape[0]
            pred = prob.argmax(dim=-1).view(-1).to(torch.int32)
            correct_num = (pred == label).sum().to(torch.float32)

        batch_all_eval_num_list,bacth_all_correct_eval_num_list = [torch.zeros(1).to(device) for _ in range(args.world_size)],[torch.zeros(1).to(device) for _ in range(args.world_size)]
        all_batch_size = torch.tensor(batch_size).to(device).to(torch.float32)
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
            print("%d/%d"%(int(all_correct_eval_num),int(all_eval_num)))

    acc = all_correct_eval_num/all_eval_num
    if rank == 0:
        print("all eval num: %f,all correct eval num: %f, acc: %f" %(all_eval_num, all_correct_eval_num, acc))
    model.train()
    return acc



dataset = load_dataset("food101", split="train[:5000]")

labels = dataset.features["label"].names
label2id, id2label = {}, {}

for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

model_checkpoint = "google/vit-base-patch16-224-in21k"
image_processor =  AutoImageProcessor.from_pretrained(model_checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose([
    RandomResizedCrop(image_processor.size["height"]),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,
])

val_transforms = Compose([
    Resize(image_processor.size["height"]),
    CenterCrop(image_processor.size["height"]),
    ToTensor(),
    normalize,
])

def preprocess_train(example_batch):
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    example_batch['pixel_values'] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

splits = dataset.train_test_split(test_size=0.1)
train_ids = splits["train"]
val_ids = splits["test"]

train_ids.set_transform(preprocess_train)
val_ids.set_transform(preprocess_val)
train_dataloader = DataLoader(train_ids, batch_size=32,shuffle=True,collate_fn=collate_fn)
val_dataloader = DataLoader(val_ids, batch_size=1,collate_fn=collate_fn)

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable parmas: {trainable_params} || all_params: {all_params} || trainable%: {100 * trainable_params / all_params}")


model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes = True
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# use accelerator to wrap


print_trainable_parameters(model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query","value"],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["classifier"]

)

lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)
optimizer = AdamW(lora_model.parameters(), lr=5e-3)

train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, val_dataloader, lora_model, optimizer
)
# train
num_epochs = 50
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

train_lora(model, num_epochs, train_dataloader,  val_dataloader, optimizer, lr_scheduler, progress_bar)


