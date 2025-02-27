# src/train_mario.py
import argparse
import os
from pathlib import Path
import importlib

import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import KFold
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset import MarioDatasetTask1
from loss import Loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--train_config",
        type=str,
        help="name of yaml config file",
        default="configs/train_mario.yaml",
    )
    return parser.parse_args()


def load_yaml_config(config_filename: str) -> dict:
    """Load yaml config.

    Args:
        config_filename: Filename to config.

    Returns:
        Loaded config.
    """
    with open(config_filename) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def make_exp_folder(config, random_state, lambda_reg):
    """Create experiment folder.

    Args:
        config: Yaml config file.

    Returns:
        Experiment folder path.
    """
    experiment_folder = os.path.join(
        "./outputs", f"{config['experiment_folder']}", f"rs_{random_state}_lambda_{lambda_reg}"
    )
    Path(experiment_folder).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(experiment_folder, "config.yaml"), "w") as outfile:
        yaml.dump(OmegaConf.to_container(config, resolve=True), outfile, default_flow_style=False)
    return experiment_folder


def train_step(model, data_loader, loss_fn, optimizer, device):
    """
    Performs one training step for the given model.

    Args:
        model (torch.nn.Module): The model to train.
        data_loader (torch.utils.data.DataLoader): DataLoader providing input data and labels.
        loss_fn (torch.nn.Module): Loss function for optimization.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        device (torch.device): Device to run the computations on (e.g., 'cuda' or 'cpu').

    Returns:
        Loss and predictions
    """
    train_loss = 0.0
    train_labels, preds, z_1os, z_1ps, z_2os, z_2ps, indices = [], [], [], [], [], [], []
    model.train()
    for i, data in enumerate(data_loader):
        image_t, image_t_next, labels, index = data
        image_t, image_t_next = image_t[:, 0:1, :, :], image_t_next[:, 0:1, :, :]
        image_t, image_t_next, labels = image_t.to(device), image_t_next.to(device), labels.to(device).long()
        train_labels.append(labels.cpu().numpy())
        indices.append(index)

        output = model(image_t, image_t_next)

        batch_alphas = alphas[index]

        if config.model_name == "OurModel":
            loss = loss_fn(output, labels, batch_alphas)
        else:
            loss = loss_fn(output, labels)
        
        if config.model_name == "OurModel":
            z_1o, z_1p, z_2o, z_2p = output
            z_1os.append(z_1o.detach().cpu().numpy())
            z_1ps.append(z_1p.detach().cpu().numpy())
            z_2os.append(z_2o.detach().cpu().numpy())
            z_2ps.append(z_2p.detach().cpu().numpy())
            preds.append(output[0].detach().cpu().numpy())
        else:
            preds.append(output.detach().cpu().numpy())
            
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)

    if config.model_name == "OurModel":
        return train_loss.item(), np.concatenate(train_labels), None, np.concatenate(indices), z_1os, z_1ps, z_2os, z_2ps
    else:
        return train_loss.item(), np.concatenate(train_labels), np.concatenate(preds), np.concatenate(indices), None, None, None, None

def val_step(data_loader, model, loss_fn, device):
    """
    Performs one evaluation step for the given model.

    Args:
        model (torch.nn.Module): The model to train.
        data_loader (torch.utils.data.DataLoader): DataLoader providing input data and labels.
        loss_fn (torch.nn.Module): Loss function for optimization.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        device (torch.device): Device to run the computations on (e.g., 'cuda' or 'cpu').

    Returns:
        Loss and predictions
    """
    test_loss = 0.0
    test_labels, preds, z_1os, z_1ps, z_2os, z_2ps, indices = [], [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            image_t, image_t_next, labels, index = data
            image_t, image_t_next = image_t[:, 0:1, :, :], image_t_next[:, 0:1, :, :]
            image_t, image_t_next, labels = image_t.to(device), image_t_next.to(device), labels.to(device).long()
            test_labels.append(labels.cpu().numpy())
            batch_alphas = alphas[index]
            indices.append(index)

            test_pred = model(image_t, image_t_next)
            if config.model_name == "OurModel":
                test_loss += loss_fn(test_pred, labels, batch_alphas)
            else:
                test_loss += loss_fn(test_pred, labels)

            if config.model_name == "OurModel":
                z_1o, z_1p, z_2o, z_2p = test_pred
                z_1os.append(z_1o.cpu().numpy())
                z_1ps.append(z_1p.cpu().numpy())
                z_2os.append(z_2o.cpu().numpy())
                z_2ps.append(z_2p.cpu().numpy())
                preds.append(test_pred[0].cpu().detach().numpy())
            else:
                preds.append(test_pred.cpu().detach().numpy())

        test_loss /= len(data_loader)
        if config.model_name == "OurModel":
            return test_loss.item(), np.concatenate(test_labels), None, np.concatenate(indices), z_1os, z_1ps, z_2os, z_2ps
        else:
            return test_loss.item(), np.concatenate(test_labels), np.concatenate(preds), np.concatenate(indices), None, None, None, None

if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(config_filename=args.train_config)
    config = OmegaConf.create(config)
    experiment_folder = make_exp_folder(config, 42, config.lambda_reg)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = importlib.import_module(f"models.{config.model_name}")
    model_class = getattr(module, config.model_name)

    dataset = MarioDatasetTask1(
        root=config.data_root_dir,
        split="train",
        valtest=False,
        )

    patient_ids = np.unique(dataset._meta["id_patient"])
    train_patient_ids, test_patient_ids = train_test_split(
        patient_ids, test_size=0.15, random_state=42
    )

    train_indices = [i for i, pid in enumerate(dataset._meta["id_patient"]) if pid in train_patient_ids]
    test_indices = [i for i, pid in enumerate(dataset._meta["id_patient"]) if pid in test_patient_ids]

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(MarioDatasetTask1(
        root=config.data_root_dir,
        split="train",
        valtest = True,
    ), test_indices)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                num_workers=20, pin_memory=True)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_patient_ids)):
        train_ids = train_patient_ids[train_idx]
        val_ids = train_patient_ids[val_idx]

        train_fold_indices = [i for i, pid in enumerate(dataset._meta["id_patient"]) if pid in train_ids]
        val_fold_indices = [i for i, pid in enumerate(dataset._meta["id_patient"]) if pid in val_ids]

        train_fold_subset = Subset(dataset, train_fold_indices)

        val_fold_subset = Subset(MarioDatasetTask1(
            root=config.data_root_dir,
            split="train",
            valtest = True,
        ), val_fold_indices)
        
        train_labels = dataset._labels[train_fold_indices]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_dataloader = DataLoader(train_fold_subset, batch_size=config.batch_size, sampler = sampler, num_workers=20, pin_memory=True)

        val_dataloader = DataLoader(val_fold_subset, batch_size=config.batch_size, shuffle=False, num_workers=20, pin_memory=True)


        backbone_name = config.backbone
        backbone = getattr(torchvision.models, backbone_name)(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        model = model_class(backbone = backbone).to(device)

        alphas = torch.nn.Parameter(torch.zeros(len(dataset), dtype=torch.float, device=device, requires_grad=True))

        param_group = [
            {"params": model.parameters(), "lr": 1e-4},
        ]
        optimizer = optim.AdamW(param_group)
        if config.lambda_reg < 10:
            optimizer.add_param_group({"params": [alphas], "lr": 1e-2})

        if config.model_name == "OurModel":
            loss_fn = Loss(lambda_reg = config.lambda_reg).to(device)
        else:
            loss_fn = nn.CrossEntropyLoss().to(device)

        val_losses = []
        for epoch in range(config.num_epochs):
            train_loss, train_labels, labels_pred_train, indices_train, z_1os, z_1ps, z_2os, z_2ps = train_step(
                data_loader=train_dataloader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device
            )

            val_loss, val_labels, labels_pred_val, indices_val, z_1os_val, z_1ps_val, z_2os_val, z_2ps_val = val_step(
                data_loader=val_dataloader,
                model=model,
                loss_fn=loss_fn,
                device=device,
            )
            
            val_losses.append(val_loss)
            
            best_model_path = os.path.join(experiment_folder, f"best_fold_{fold + 1}.pth")

            if val_loss == min(val_losses):
                torch.save(model.state_dict(), best_model_path)
                best_model_state = model.state_dict()
                if config.model_name == "OurModel":
                    np.save(os.path.join(experiment_folder, f"alphas_fold_{fold + 1}"), alphas.cpu().detach().numpy())
        
        model.load_state_dict(best_model_state)

        test_loss, test_labels, labels_pred_test, indices, z_1os, z_1ps, z_2os, z_2ps = val_step(
            data_loader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            device=device,
        )

        np.save(os.path.join(experiment_folder, f"labels_pred_test_fold_{fold + 1}"), labels_pred_test)
        if config.model_name == "OurModel":
            np.save(os.path.join(experiment_folder, f"z_1os_test_fold_{fold + 1}"), np.concatenate(z_1os))
            np.save(os.path.join(experiment_folder, f"z_1ps_test_fold_{fold + 1}"), np.concatenate(z_1ps))
            np.save(os.path.join(experiment_folder, f"z_2os_test_fold_{fold + 1}"), np.concatenate(z_2os))
            np.save(os.path.join(experiment_folder, f"z_2ps_test_fold_{fold + 1}"), np.concatenate(z_2ps))
        np.save(os.path.join(experiment_folder, f"labels_test"), test_labels)
        np.save(os.path.join(experiment_folder, f"indices_test"), indices)

        train_dataloader = DataLoader(train_fold_subset, batch_size=config.batch_size, shuffle=False, num_workers=20, pin_memory=True)

        train_loss, train_labels, labels_pred_train, indices_train, z_1os, z_1ps, z_2os, z_2ps = val_step(
            data_loader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            device=device,
        )

        if config.model_name == "OurModel":
            np.save(os.path.join(experiment_folder, f"z_1os_train_fold_{fold + 1}"), np.concatenate(z_1os))
            np.save(os.path.join(experiment_folder, f"z_1ps_train_fold_{fold + 1}"), np.concatenate(z_1ps))
            np.save(os.path.join(experiment_folder, f"z_2os_train_fold_{fold + 1}"), np.concatenate(z_2os))
            np.save(os.path.join(experiment_folder, f"z_2ps_train_fold_{fold + 1}"), np.concatenate(z_2ps))
        np.save(os.path.join(experiment_folder, f"labels_pred_train_fold_{fold + 1}"), labels_pred_train)
        np.save(os.path.join(experiment_folder, f"indices_train_fold_{fold + 1}"), indices_train)
        np.save(os.path.join(experiment_folder, f"labels_train_fold_{fold + 1}"), train_labels)