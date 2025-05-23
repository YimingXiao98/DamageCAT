import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils

import data_config
from datasets.CD_dataset import CDDataset, xBDataset, DamageCATDataset


# Loader for testing
def get_loader(
    data_name,
    img_size=256,
    batch_size=8,
    split="test",
    is_train=False,
    dataset="DamageCATDataset",
    patch=None,
    random_seed=42,
):
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    print(dataConfig)

    if dataset == "CDDataset":
        data_set = CDDataset(
            root_dir=root_dir,
            split=split,
            img_size=img_size,
            is_train=is_train,
            label_transform=label_transform,
            patch=patch,
            random_seed=random_seed,
        )
    elif dataset == "xBDataset":
        data_set = xBDataset(
            root_dir=root_dir,
            split=split,
            img_size=img_size,
            is_train=is_train,
            label_transform=label_transform,
            random_seed=random_seed,
        )
    elif dataset == "DamageCATDataset":
        data_set = DamageCATDataset(
            root_dir=root_dir,
            split=split,
            img_size=img_size,
            is_train=is_train,
            label_transform=label_transform,
            random_seed=random_seed,
        )
    else:
        raise NotImplementedError(
            "Wrong dataset name %s (choose one from [CDDataset])" % dataset
        )

    shuffle = is_train
    dataloader = DataLoader(
        data_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return dataloader


def get_loaders(args):

    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = "val"
    if hasattr(args, "split_val"):
        split_val = args.split_val
    if args.dataset == "CDDataset":
        training_set = CDDataset(
            root_dir=root_dir,
            split=split,
            img_size=args.img_size,
            is_train=True,
            label_transform=label_transform,
            random_seed=args.random_seed,
        )
        val_set = CDDataset(
            root_dir=root_dir,
            split=split_val,
            img_size=args.img_size,
            is_train=False,
            label_transform=label_transform,
            random_seed=args.random_seed,
        )
    elif args.dataset == "xBDataset":
        training_set = xBDataset(
            root_dir=root_dir,
            split=split,
            img_size=args.img_size,
            is_train=True,
            label_transform=label_transform,
            random_seed=args.random_seed,
        )
        val_set = xBDataset(
            root_dir=root_dir,
            split=split_val,
            img_size=args.img_size,
            is_train=False,
            label_transform=label_transform,
            random_seed=args.random_seed,
        )
    elif args.dataset == "DamageCATDataset":
        training_set = DamageCATDataset(
            root_dir=root_dir,
            split=split,
            img_size=args.img_size,
            is_train=True,
            label_transform=label_transform,
            random_seed=args.random_seed,
        )
        val_set = DamageCATDataset(
            root_dir=root_dir,
            split=split_val,
            img_size=args.img_size,
            is_train=False,
            label_transform=label_transform,
            random_seed=args.random_seed,
        )
    else:
        raise NotImplementedError(
            "Wrong dataset name %s (choose one from [CDDataset,])" % args.dataset
        )
    # Check the first item of the dataset
    print(f"Dataset Type: {type(training_set)}")
    print(f"First item in dataset: {training_set[0]['L'].shape}")
    print(
        f"Min Max value in label: {training_set[0]['L'].max(), training_set[0]['L'].min()}"
    )
    print(
        f"Min Max value in A: {training_set[0]['A'].max(), training_set[0]['A'].min()}"
    )
    print(
        f"Min Max value in B: {training_set[0]['B'].max(), training_set[0]['B'].min()}"
    )

    datasets = {"train": training_set, "val": val_set}
    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        for x in ["train", "val"]
    }
    return dataloaders


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    # tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(",")
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
