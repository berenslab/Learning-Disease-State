# src/train_internal.py
import os
import torch
import torchvision
import numpy as np
import argparse
import yaml
from omegaconf import OmegaConf
from pathlib import Path

from models import OurModel
from torch import nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--train_config",
        type=str,
        help="name of yaml config file",
        default="configs/train_internal.yaml",
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

def make_exp_folder(config):
    """Create experiment folder.

    Args:
        config: Yaml config file.

    Returns:
        Experiment folder path.
    """
    experiment_folder = os.path.join(
        "./outputs", f"{config['experiment_folder']}", f"shots"
    )
    Path(experiment_folder).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(experiment_folder, "config.yaml"), "w") as outfile:
        yaml.dump(OmegaConf.to_container(config, resolve=True), outfile, default_flow_style=False)
    return experiment_folder

def train_naive():
    directory = config.dir_naive
    bal_acc = []
    for fold in range(5):
        backbone = getattr(torchvision.models, "resnet50")(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        checkpoint = torch.load(directory + f"best_fold_{fold+1}.pth", map_location=device, weights_only=True)
        backbone_state_dict = {k: v for k, v in checkpoint.items() if k.startswith("backbone.")}
        backbone.load_state_dict(backbone_state_dict, strict=False)
        backbone = backbone.to(device)
        backbone.eval()

        train_features, train_labels = [], []

        for images, labels in dataloader_train:
            images = images.to(device)
            with torch.no_grad():
                features = backbone(images[:, 0:1, :, :].to(device))
            train_features.append(features.cpu().numpy())
            train_labels.extend([0 if label == "N" else 1 for label in labels])
        
        train_features = np.vstack(train_features)
        train_labels = np.array(train_labels)

        clf = LogisticRegression(random_state=42, max_iter=10000).fit(train_features, train_labels)

        labels = []
        labels_pred = []
        for idx, batch in enumerate(dataloader):
            images, label = batch
            images = images.to(device)
            labels.append(label)
            with torch.no_grad():
                features = backbone(images[:, 0:1, :, :].to(device))
            labels_pred.append(clf.predict(features.cpu().numpy()))
        labels_pred = np.concatenate(labels_pred)
        labels = np.concatenate(labels)
        labels = np.array([0 if label == "N" else 1 for label in labels]) 
        bal_acc.append(balanced_accuracy_score(labels, labels_pred))
    return labels, labels_pred, bal_acc

def train_db(directory):
    thresholds = []
    for fold in range(5) :
        backbone = getattr(torchvision.models, "resnet50")(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        model = OurModel.OurModel(backbone = backbone)
        model.load_state_dict(torch.load(directory + "best_fold_" + str(fold+1) + ".pth", weights_only=True))

        model.eval()
        model = model.to(device)

        labels = []
        z_ps = []
        for idx, batch in enumerate(dataloader_train):
            images, label = batch
            labels.append(label)
            images = images.to(device)
            with torch.no_grad():
                features = model(images[:, 0:1, :, :], images[:, 0:1, :, :])
            z_ps.append(features[1])
        z_ps = torch.cat(z_ps, dim=0).squeeze().cpu().numpy()
        labels = np.concatenate(labels)

        X = z_ps
        y = np.array([0 if label == "N" else 1 for label in labels])

        sorted_indices = np.argsort(X)
        X_sorted = X[sorted_indices]

        best_accuracy = -0.1
        best_threshold = None
        for i in range(1, len(X_sorted)):
            threshold = (X_sorted[i - 1] + X_sorted[i]) / 2

            y_pred = (X >= threshold).astype(int)

            accuracy = balanced_accuracy_score(y, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        thresholds.append(best_threshold)

    bal_acc = []
    z_pss = []
    for fold in range(5) :
        backbone = getattr(torchvision.models, "resnet50")(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        model = OurModel.OurModel(backbone = backbone)
        model.load_state_dict(torch.load(directory + "best_fold_" + str(fold+1) + ".pth", weights_only=True))

        model.eval()
        model = model.to(device)

        z_ps = []
        labels = []
        for idx, batch in enumerate(dataloader):
            images, label = batch
            labels.append(label)
            images = images.to(device)
            with torch.no_grad():
                features = model(images[:, 0:1, :, :], images[:, 0:1, :, :])
            z_ps.append(features[1])
        z_ps = torch.cat(z_ps).squeeze().cpu().numpy()
        labels = np.concatenate(labels)
        labels = np.array([0 if label == "N" else 1 for label in labels])

        y_pred = (z_ps >= thresholds[fold]).astype(int)

        z_pss.append(z_ps)
        bal_acc.append(balanced_accuracy_score(labels, y_pred))

    return z_pss, labels, bal_acc

if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(config_filename=args.train_config)
    config = OmegaConf.create(config)
    experiment_folder = make_exp_folder(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = your_dataset(    
        ## Load your pytorch dataset for evaluation
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=20,
        pin_memory=True,
    )

    L_n_shots = [1, 2, 4, 8, 16, 32, 64, 128]

    for n_shots in L_n_shots:

        dataset_train = your_dataset_train(
            ## Load your pytorch dataset for training (balanced samples)
        )

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=5,
            pin_memory=True,
        )
  
        all_those_balanced_accuracies = []

        labels_naive = []
        labels_pred_naive = []

        labels, labels_pred, bal_acc = train_naive()

        labels_naive.append(labels)
        labels_pred_naive.append(labels_pred)
        all_those_balanced_accuracies.append(bal_acc)


        all_z_p_no_gamma, labels_no_gamma = [], []

        z_pss, labels, bal_acc = train_db(directory = config.dir_no_gamma)

        all_z_p_no_gamma.append(z_pss)
        labels_no_gamma.append(labels)

        all_those_balanced_accuracies.append(bal_acc)

        all_z_p_gamma = []
        labels_gamma = []

        z_pss, labels, bal_acc = train_db(directory = config.dir_gamma)

        all_z_p_gamma.append(z_pss)
        labels_gamma.append(labels)

        all_those_balanced_accuracies.append(bal_acc)

        np.save(experiment_folder + f"{n_shots}_shots_z_p_gamma.npy", np.array(all_z_p_gamma))
        np.save(experiment_folder + f"{n_shots}_shots_z_p_no_gamma.npy", np.array(all_z_p_no_gamma))
        np.save(experiment_folder + f"{n_shots}_shots_labels_gamma.npy", np.array(labels_gamma))
        np.save(experiment_folder + f"{n_shots}_shots_labels_no_gamma.npy", np.array(labels_no_gamma))
        np.save(experiment_folder  + f"{n_shots}_shots_bal_acc.npy", np.array(all_those_balanced_accuracies))
        np.save(experiment_folder + f"{n_shots}_shots_labels_naive.npy", np.array(labels_naive))
        np.save(experiment_folder + f"{n_shots}_shots_labels_pred_naive.npy", np.array(labels_pred_naive))