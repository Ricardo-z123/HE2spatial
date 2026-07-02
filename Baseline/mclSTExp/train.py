import argparse
import torch
import os
import random
import numpy as np
from dataset import HERDataset
from model import mclSTExp_Attention
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AvgMeter, get_lr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--max_epochs', type=int, default=30, help='')
    parser.add_argument('--temperature', type=float, default=1., help='temperature')
    parser.add_argument('--fold', type=int, default=0, help='fold')
    parser.add_argument('--dim', type=int, default=785, help='spot_embedding dimension (# HVGs)')
    parser.add_argument('--image_embedding_dim', type=int, default=1024, help='image_embedding dimension')
    parser.add_argument('--projection_dim', type=int, default=256, help='projection_dim ')
    parser.add_argument('--heads_num', type=int, default=8, help='attention heads num')
    parser.add_argument('--heads_dim', type=int, default=64, help='attention heads dim')
    parser.add_argument('--heads_layers', type=int, default=2, help='attention heads layer num')
    parser.add_argument('--dropout', type=float, default=0., help='dropout')
    parser.add_argument('--encoder_name', type=str, default='densenet121', help='image encoder')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    return args


def train(model, train_dataLoader, optimizer, epoch):
    loss_meter = AvgMeter()
    tqdm_train = tqdm(train_dataLoader, total=len(train_dataLoader))
    
    # for batch in tqdm_train:
    #     batch = {k: v.cuda() for k, v in batch.items() if
    #              k == "image" or k == "expression" or k == "position"}
    #     loss = model(batch)
    #     optimizer.zero_grad()
        
    for batch in tqdm_train:
        if tqdm_train.n == 0:
            from collections import Counter
            slide_ids = batch["slide_id"].tolist()
            counter = Counter(slide_ids)
            id2name = train_dataLoader.dataset.id2name
            slide_names = {id2name[k]: v for k, v in sorted(counter.items())}
            print(f"\n[Epoch {epoch}] batch slide distribution ({len(slide_ids)}spots from {len(counter)} slides):")
            print(slide_names)
        batch = {k: v.cuda() for k, v in batch.items() if
                 k == "image" or k == "expression" or k == "position"}
        loss, spots_loss, images_loss = model(batch)
        if tqdm_train.n == 1:
            print(f"[Epoch {epoch}] spots_loss={spots_loss.item():.4f}  "
                  f"images_loss={images_loss.item():.4f}  "
                  f"diff={abs(spots_loss.item()-images_loss.item()):.6f}")
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_train.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer), epoch=epoch)


def load_data(args):
    print('load dataset: her2st')
    train_dataset = HERDataset(train=True, fold=args.fold)
    train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = HERDataset(train=False, fold=args.fold)
    return train_dataLoader, test_dataset


def save_model(args, model, test_dataset):
    os.makedirs(f"/root/autodl-tmp/mclSTExp/model_result/her2st/{test_dataset.id2name[0]}", exist_ok=True)
    torch.save(model.state_dict(),
               f"/root/autodl-tmp/mclSTExp/model_result/her2st/{test_dataset.id2name[0]}/best_{args.fold}.pt")


def main():
    args = generate_args()
    set_seed(args.seed)
    for i in range(6):
        args.fold = i
        print("当前fold:", args.fold)
        train_dataLoader, test_dataset = load_data(args)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = mclSTExp_Attention(encoder_name=args.encoder_name,
                                   spot_dim=args.dim,
                                   temperature=args.temperature,
                                   image_dim=args.image_embedding_dim,
                                   projection_dim=args.projection_dim,
                                   heads_num=args.heads_num,
                                   heads_dim=args.heads_dim,
                                   head_layers=args.heads_layers,
                                   dropout=args.dropout)
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-4, weight_decay=1e-3
        )
        for epoch in range(args.max_epochs):
            model.train()
            train(model, train_dataLoader, optimizer, epoch)

        save_model(args, model, test_dataset)
        print("Saved Model")


if __name__ == '__main__':
    main()
