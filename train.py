import os
import random
from argparse import ArgumentParser
from datetime import datetime
from importlib import import_module

import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import PDL1Dataset, Physio2019Dataset
from model import SimtaPlus
from utils import logger


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_cmd_args():
    parser = ArgumentParser()
    parser.add_argument("--task", required=True,
        choices=["physionet", "pdl1"])
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--include_oi", action="store_true")
    parser.add_argument("--batch_size", default=128, type=int)   # 8-64
    parser.add_argument("--temp_c", default=64, type=int)   #  12-128
    parser.add_argument("--stat_c", default=64, type=int)
    parser.add_argument("--attn_c", default=16, type=int)   #  4-32
    parser.add_argument("--layers", default=3, type=int)    # 1/2/3
    parser.add_argument("--dropout", default=0.2, type=float)    # 0-0.5
    parser.add_argument("--weight_decay", default=1e-2, type=float)   #
    parser.add_argument("--max_lr", default=1e-3, type=float)   # e-5 e-1
    parser.add_argument("--min_lr", default=1e-6, type=float)   # e-8 e-5
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--wpos", default=0.5, type=float)
    parser.add_argument("--seq_drop_prob", default=0, type=float)
    parser.add_argument("--seq_drop_pct", default=0.2, type=float)
    parser.add_argument("--sgd_pct", default=0.5, type=float)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()

    return args


def init_dataloaders(args):
    df = pd.read_csv(cfg.DF_PATH)
    if args.task == "physionset" and args.include_oi:
        oi_stats = pd.read_csv(cfg.OI_STATS_PATH)

    df_train = df.loc[df.subset == "train"].reset_index(drop=True)
    df_val = df.loc[df.subset == "val"].reset_index(drop=True)
    df_test = df.loc[df.subset == "test"].reset_index(drop=True)
    if args.task == "physionet":
        if args.include_oi:
            ds_train = Physio2019Dataset(df_train, oi_stats, args.seq_drop_prob,
                args.seq_drop_pct)
            ds_val = Physio2019Dataset(df_val, oi_stats)
            ds_test = Physio2019Dataset(df_test, oi_stats)
        else:
            ds_train = Physio2019Dataset(df_train,
                seq_drop_prob=args.seq_drop_prob, seq_drop_pct=args.seq_drop_pct)
            ds_val = Physio2019Dataset(df_val)
            ds_test = Physio2019Dataset(df_test)
        dl_train = Physio2019Dataset.get_dataloader(ds_train, args.batch_size,
            True, args.workers)
        dl_val = Physio2019Dataset.get_dataloader(ds_val, args.batch_size,
            args.workers)
        dl_test = Physio2019Dataset.get_dataloader(ds_test, args.batch_size,
            args.workers)
    else:
        ds_train = PDL1Dataset(df_train)
        ds_val = PDL1Dataset(df_val)
        ds_test = PDL1Dataset(df_test)
        dl_train = PDL1Dataset.get_dataloader(ds_train, args.batch_size,
            True, args.workers)
        dl_val = PDL1Dataset.get_dataloader(ds_val, args.batch_size,
            args.workers)
        dl_test = PDL1Dataset.get_dataloader(ds_test, args.batch_size,
            args.workers)

    return dl_train, dl_val, dl_test


def init_models(args, dl_train_len):
    temp_in_channels = cfg.TEMP_IN_CHANNELS
    stat_in_channels = cfg.STAT_IN_CHANNELS
    if args.include_oi:
        stat_in_channels += cfg.TEMP_IN_CHANNELS
        temp_in_channels += cfg.TEMP_IN_CHANNELS
    model = SimtaPlus(
        temp_in_channels=temp_in_channels,
        stat_in_channels=stat_in_channels,
        temp_hidden_channels=args.temp_c,
        stat_hidden_channels=args.stat_c,
        attn_hidden_channels=args.attn_c,
        num_layers=args.layers,
        num_classes=cfg.NUM_CLASSES,
        drop_prob=args.dropout
    ).cuda()

    class_weights = torch.tensor([1 - args.wpos, args.wpos],
        dtype=torch.float).cuda()
    criterion = nn.CrossEntropyLoss(class_weights)

    optimizer = optim.AdamW(model.parameters(), args.max_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
        args.epochs * dl_train_len, args.min_lr)

    return model, criterion, optimizer, scheduler


def init_tensorboard(args):
    if args.name == "":
        case_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        case_name = args.name
    log_dir = os.path.join(cfg.LOG_DIR, case_name)
    tb_writer = SummaryWriter(log_dir)

    return tb_writer, log_dir


@logger
def train_epoch(model, dl, criterion, optimizer, scheduler, best_score_val, log_dir):
    model.train()
    y_true = []
    y_pred = []
    avg_loss = 0

    progress = tqdm(total=len(dl))
    for _, sample in enumerate(dl):
        optimizer.zero_grad()

        x_temp, x_stat, y_full, y_max, t, masks, lengths = sample
        x_temp = x_temp.cuda()
        x_stat = x_stat.cuda()
        t = t.cuda()
        masks = masks.cuda()
        targets = y_max.cuda()
        lengths = lengths.cuda()

        output, output_max = model(x_temp, x_stat, t, masks, lengths)
        loss = criterion(output_max, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        avg_loss += loss.cpu().item()

        y_true.append(y_max.numpy())
        y_pred.append(output_max.detach().cpu().numpy()[:, 1])

        progress.set_postfix_str(f"loss={loss:.4f}")
        progress.update()
    progress.close()

    avg_loss /= len(dl)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    score = roc_auc_score(y_true, y_pred)

    if score > best_score_val:
        result = {
            'y_true':y_true,
            'y_pred':y_pred
        }
        pd.DataFrame(result).to_csv(os.path.join(log_dir, "best_score.csv"))

    return avg_loss, score


@logger
@torch.no_grad()
def eval_epoch(model, dl, criterion):
    model.eval()
    y_true = []
    y_pred = []
    avg_loss = 0

    for _, sample in enumerate(dl):
        x_temp, x_stat, y_full, y_max, t, masks, lengths = sample
        x_temp = x_temp.cuda()
        x_stat = x_stat.cuda()
        t = t.cuda()
        masks = masks.cuda()
        targets = y_max.cuda()
        lengths = lengths.cuda()

        output, output_max = model(x_temp, x_stat, t, masks, lengths)
        loss = criterion(output_max, targets)

        avg_loss += loss.cpu().item()

        y_true.append(y_max.numpy())
        y_pred.append(output_max.detach().cpu().numpy()[:, 1])

    avg_loss /= len(dl)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    score = roc_auc_score(y_true, y_pred)

    return avg_loss, score


def log_tensorboard(tb_writer, losses, metrics, epoch):
    tb_writer.add_scalars("loss", losses, epoch)
    tb_writer.add_scalars("AUC", metrics, epoch)
    tb_writer.flush()


def print_logs(losses, metrics):
    print(f"Loss = {losses['train']:.4f} | {losses['val']:.4f} | {losses['test']:.4f}")
    print(f"AUC = {metrics['train']:.4f} | {metrics['val']:.4f} | {metrics['test']:.4f}")


def log_hyperparams(tb_writer, args, metrics):
    hyper_params = {k: v for k, v in args.__dict__.items()
        if k not in ["gpu", "name", "workers"]}
    tb_writer.add_hparams(hyper_params, metrics)
    tb_writer.flush()


def main():
    set_rng_seed(42)

    args = parse_cmd_args()
    global cfg
    cfg = import_module(f"configs.{args.task}_configs")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dl_train, dl_val, dl_test = init_dataloaders(args)

    model, criterion, optimizer, scheduler = init_models(args, len(dl_train))

    best_score = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    tb_writer, log_dir = init_tensorboard(args)
    for i in range(args.epochs):
        print(f"Epoch {i}")

        if i == int(args.sgd_pct * args.epochs):
            cur_lr = optimizer.state_dict()["param_groups"][0]["lr"]
            optimizer = optim.SGD(model.parameters(), cur_lr, 0.9, weight_decay=args.weight_decay)
            scheduler_state = scheduler.state_dict()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                args.epochs * len(dl_train), args.min_lr)
            scheduler.load_state_dict(scheduler_state)

        loss_train, score_train = train_epoch(model, dl_train, criterion,
            optimizer, scheduler, best_score['val'], log_dir)

        loss_val, score_val = eval_epoch(model, dl_val, criterion)
        loss_test, score_test = eval_epoch(model, dl_test, criterion)

        losses = {
            "train": loss_train,
            "val": loss_val,
            "test": loss_test
        }
        metrics = {
            "train": score_train,
            "val": score_val,
            "test": score_test
        }
        log_tensorboard(tb_writer, losses, metrics, i)

        if score_val >= best_score["val"]:
            best_score["train"] = score_train
            best_score["val"] = score_val
            best_score["test"] = score_test
            torch.save(model.state_dict(),
                os.path.join(log_dir, "best_model.pth"))

        print_logs(losses, metrics)

    log_hyperparams(tb_writer, args, best_score)


if __name__ == "__main__":
    main()
