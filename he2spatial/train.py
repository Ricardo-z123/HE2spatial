import os
from tqdm import tqdm

import torch
from torch import nn
import torch.distributed as dist
import torch.utils.data.distributed

import config as CFG
from dataset import CLIPDataset
from models import CLIPModel, CLIPModel_ViT, CLIPModel_ViT_L, CLIPModel_CLIP, CLIPModel_resnet101, \
    CLIPModel_resnet152
from utils import AvgMeter, get_lr
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='DDP for CLIP')

parser.add_argument('--exp_name', type=str, default='clip', help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--max_epochs', type=int, default=4, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
# parser.add_argument('--dist-backend', default='gloo', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')

parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
parser.add_argument('--model', type=str, default='resnet50', help='')


def build_loaders(args):
    # slice 3 randomly chosen to be test and will be left out during training
    print("Building loaders")
    dataset = CLIPDataset(image_path="/root/autodl-tmp/pics/P1.tiff",
                          spatial_pos_path="EGA-dataset/data/tissue_pos_matrices/tissue_positions_list_1.csv",
                          reduced_mtx_path="EGA-dataset/data/filtered_expression_matrices/1/harmony_matrix.npy",
                          barcode_path="EGA-dataset/data/filtered_expression_matrices/1/barcodes.tsv")
    dataset2 = CLIPDataset(image_path="/root/autodl-tmp/pics/P2.tiff",
                           spatial_pos_path="EGA-dataset/data/tissue_pos_matrices/tissue_positions_list_2.csv",
                           reduced_mtx_path="EGA-dataset/data/filtered_expression_matrices/2/harmony_matrix.npy",
                           barcode_path="EGA-dataset/data/filtered_expression_matrices/2/barcodes.tsv")
    dataset3 = CLIPDataset(image_path="/root/autodl-tmp/pics/P3.tiff",
                           spatial_pos_path="EGA-dataset/data/tissue_pos_matrices/tissue_positions_list_3.csv",
                           reduced_mtx_path="EGA-dataset/data/filtered_expression_matrices/3/harmony_matrix.npy",
                           barcode_path="EGA-dataset/data/filtered_expression_matrices/3/barcodes.tsv")
    dataset5 = CLIPDataset(image_path="/root/autodl-tmp/pics/P5.tiff",
                           spatial_pos_path="EGA-dataset/data/tissue_pos_matrices/tissue_positions_list_5.csv",
                           reduced_mtx_path="EGA-dataset/data/filtered_expression_matrices/5/harmony_matrix.npy",
                           barcode_path="EGA-dataset/data/filtered_expression_matrices/5/barcodes.tsv")
    dataset6 = CLIPDataset(image_path="/root/autodl-tmp/pics/P6.tiff",
                           spatial_pos_path="EGA-dataset/data/tissue_pos_matrices/tissue_positions_list_6.csv",
                           reduced_mtx_path="EGA-dataset/data/filtered_expression_matrices/6/harmony_matrix.npy",
                           barcode_path="EGA-dataset/data/filtered_expression_matrices/6/barcodes.tsv")
    dataset7 = CLIPDataset(image_path="/root/autodl-tmp/pics/P7.tiff",
                           spatial_pos_path="EGA-dataset/data/tissue_pos_matrices/tissue_positions_list_7.csv",
                           reduced_mtx_path="EGA-dataset/data/filtered_expression_matrices/7/harmony_matrix.npy",
                           barcode_path="EGA-dataset/data/filtered_expression_matrices/7/barcodes.tsv")
    dataset8 = CLIPDataset(image_path="/root/autodl-tmp/pics/P8.tiff",
                           spatial_pos_path="EGA-dataset/data/tissue_pos_matrices/tissue_positions_list_8.csv",
                           reduced_mtx_path="EGA-dataset/data/filtered_expression_matrices/8/harmony_matrix.npy",
                           barcode_path="EGA-dataset/data/filtered_expression_matrices/8/barcodes.tsv")

    dataset = torch.utils.data.ConcatDataset([dataset, dataset2, dataset3, dataset5, dataset6, dataset7, dataset8])

    # ---------- train / test split (per-slide) ---------- #
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(42))
    print(len(train_dataset), len(test_dataset))
    print("train/test split completed")

    # Set up distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)  # by default, rank and world sizes are retrieved from env variables
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, drop_last=True)

    print("Finished building loaders")
    return train_loader, test_loader


def cleanup():
    dist.destroy_process_group()


def train_epoch(model, train_loader, optimizer, args, lr_scheduler=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:

        batch = {k: v.cuda() for k, v in batch.items()
                 if k in ["image", "reduced_expression", "positions"]}  ## 修改1
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()

        # 修改
        for param in model.parameters():
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= args.world_size

        optimizer.step()
        # if step == "batch":
        #   lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter


def test_epoch(model, test_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items()
                 if k in ["image", "reduced_expression", "positions"]}  ## 修改2
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    print("Starting...")
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    current_device = local_rank
    torch.cuda.set_device(current_device)

    """ this block initializes a process group and initiate communications
		between all processes running on all nodes """

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    # init the process group
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size,
                            rank=rank)
    print("process group ready!")

    # make the model
    print('From Rank: {}, ==> Making model..'.format(rank))
    if args.model == "clip":
        model = CLIPModel_CLIP().cuda(current_device)
        print("Image encoder is CLIP")
    elif args.model == "vit":
        model = CLIPModel_ViT().cuda(current_device)
        print("Image encoder is ViT")
    elif args.model == "vit_l":
        model = CLIPModel_ViT_L().cuda(current_device)
        print("Image encoder is ViT_L")
    elif args.model == "resnet101":
        model = CLIPModel_resnet101().cuda(current_device)
        print("Image encoder is ResNet101")
    elif args.model == "resnet152":
        model = CLIPModel_resnet152().cuda(current_device)
        print("Image encoder is ResNet152")
    else:
        model = CLIPModel().cuda(current_device)
        print("Image encoder is ResNet50")
    model = nn.parallel.DistributedDataParallel(model, device_ids=[current_device])

    # load the data
    print('From Rank: {}, ==> Preparing data..'.format(rank))
    train_loader, test_loader = build_loaders(args)

    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    # )

    # # # SGD
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=CFG.lr*10,
    #     momentum=0.9,
    #     weight_decay=CFG.weight_decay,
    #     nesterov=True
    # )
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=args.max_epochs, eta_min=1e-6
    # )

    # Train the model for a fixed number of epochs
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.max_epochs):
        print(f"Epoch: {epoch + 1}")
        # step = "epoch"

        train_loader.sampler.set_epoch(epoch)

        # Train the model
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, args)

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(model, test_loader)

        if test_loss.avg < best_loss and rank == 0:
            if not os.path.exists(str(args.exp_name)):
                os.mkdir(str(args.exp_name))
            best_loss = test_loss.avg
            best_epoch = epoch

            torch.save(model.state_dict(), str(args.exp_name) + "/best.pt")
            print("Saved Best Model! Loss: {}".format(best_loss))

    print("Done!, final loss: {}".format(best_loss))
    print("Best epoch: {}".format(best_epoch))
    cleanup()


if __name__ == "__main__":
    main()

