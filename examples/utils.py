import random
import numpy as np
import torch

from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from models import SimpleNet, resnet18, vgg16_bn


def set_seed(seed: int, device)->None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

def get_model(model_name: str) -> torch.nn.Module:
    model = None

    prepared_models = {
        "simple_model": SimpleNet(),
        "resnet18": resnet18(num_classes=10),
        "vgg16": vgg16_bn(num_classes=10)
    }

    model = prepared_models.get(model_name)

    if model is None:
        raise ValueError(f"model '{model_name}' not found in prepared models")

    return model


def calc_acc(y: Tensor, t: Tensor) -> Tensor:
    return torch.sum(torch.argmax(y, dim=1) == t)

def setup_dateset(hparams):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    train_batch_size = hparams.batch_size
    test_batch_size = hparams.batch_size

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    
    return trainloader, testloader, train_batch_size, test_batch_size

def test(model, testloader, criterion, device, optimizer):
    test_dataset_size = len(testloader.dataset)
    model.eval()
    acc = 0.0
    loss = 0.0
    optimizer.sample_params(sampling=False)
    with torch.inference_mode():
        for i, data in enumerate(testloader):
            x, t = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            y = model(x)
            loss += criterion(y, t)
            acc += calc_acc(y, t)
        loss /= test_dataset_size
        acc /= test_dataset_size
        acc *= 100
    return loss, acc

def save_params_dir(save_dir:str, model) -> str:
    return f"{save_dir}/{model}"


def calc_ece(outputs, targets, bin_size=10) -> dict:
    class_pred = np.argmax(outputs, axis=1)
    conf = np.max(outputs, axis=1)

    # Storage
    acc_tab = np.zeros(bin_size)  # empirical (true) confidence
    mean_conf = np.zeros(bin_size)  # predicted confidence
    nb_items_bin = np.zeros(bin_size)  # number of items in the bins

    # tau_tab : A list dealing with a range of confidence. [0, 1] divided by the number of bin_size
    tau_tab = np.linspace(0, 1, bin_size + 1)

    freqs = np.zeros(bin_size)

    for i in np.arange(bin_size):
        # Select those whose confidence is in the range (tau_tab[i], tau_tab[i+1])
        sec = (tau_tab[i] < conf) & (conf <= tau_tab[i + 1])

        # Count the number of items for which confidence is in the range [tau_tab[i], tau_tab[i+1]].
        nb_items_bin[i] = np.sum(sec)

        # Extract the predicted class and the true class (label)
        class_pred_sec, y_sec = class_pred[sec], targets[sec]

        # If the number of confidences in [tau_tab[i], tau_tab[i+1]] is greater than zero, compute the average of the confidences in the range. Otherwise, return np.nan.
        mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan

        # compute the empirical confidence
        acc_tab[i] = np.mean(
            class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

        freqs[i] = sec.sum()

    # Cleaning
    mean_conf[np.isnan(mean_conf)] = 0
    acc_tab[np.isnan(acc_tab)] = 0
    nb_items_bin[np.isnan(nb_items_bin)] = 0
    # mean_conf = mean_conf[nb_items_bin > 0]
    # acc_tab = acc_tab[nb_items_bin > 0]
    # nb_items_bin = nb_items_bin[nb_items_bin > 0]

    # Reliability diagram
    reliability_diag = {"mean_conf": mean_conf,
                        "tau_tab": tau_tab, "acc_tab": acc_tab}
    # Expected Calibration Error
    ece = np.average(
        np.absolute(mean_conf - acc_tab),
        weights=nb_items_bin.astype(float) / np.sum(nb_items_bin),
    )
    # Maximum Calibration Error
    mce = np.max(np.absolute(mean_conf - acc_tab))

    # Saving
    return ece, mce, freqs, reliability_diag, conf
