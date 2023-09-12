import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from typing import Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

### original ###
from optim import AdamB
from utils import set_seed, calc_acc, setup_dateset, get_model, calc_ece, save_params_dir
from args import parse

def main():

    hparams = parse()

    #### deterministic ###
    if hparams.is_debug:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    set_seed(seed=hparams.inference_seed, device=device)

    load_dir = save_params_dir(hparams.save_dir, hparams.model)
    print(f'load_dir: {load_dir}')

    model = get_model(hparams.model)
    model = model.to(device)
    print(f'model: {hparams.model}')

    ### need to load model for non bayesian params ###
    model.load_state_dict(torch.load(os.path.join(load_dir, 'model.pth'), map_location=torch.device("cpu")))

    ### setup AdamB ###
    opt_params = {
        "device": device,
    }
    optimizer = AdamB(model, **opt_params)
    optimizer.load_state_dict(torch.load(os.path.join(load_dir, 'optimizer.pth'), map_location=torch.device("cpu")))
    
    inference_dir = os.path.join(load_dir, 'inference')
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    model.eval()

    trainloader, testloader, train_batch_size, test_batch_size = setup_dateset(hparams)

    if hparams.ensemble_num == -1:
        """ use parameter mu, no use sampling """
        num_samples = 1
        sampling = False
    else:
        """ use sampling """
        num_samples = hparams.ensemble_num
        sampling = True
    
    with torch.inference_mode():
        num_test_data = len(testloader.dataset)
        num_classes = 10
        ensemble_outputs = torch.zeros((num_test_data, num_classes)).to(device)
        targets = torch.zeros(num_test_data).to(device)

        #single_outputs = torch.zeros((num_test_data, num_classes)).to(device)

        for i in range(num_samples):
            optimizer.sample_params(sampling=sampling)
            for iter, data in enumerate(testloader):
                x, t = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
                begin = iter * len(t)
                end = begin + len(t)

                y = model(x)
                #print(y)
                prob = F.softmax(y, dim=1)
                ensemble_outputs[begin : end] += prob
                targets[begin : end] = t

            if ((i+1)% 5 == 0) or (i == 0):
                mean_outputs_cpu = ensemble_outputs.detach().cpu() / (i + 1)
                targets_cpu = targets.detach().cpu()

                acc = calc_acc(mean_outputs_cpu, targets_cpu) / num_test_data
                acc *=100
                ece, _, _, _, _ = calc_ece(mean_outputs_cpu.numpy() , targets_cpu.numpy() )
                print(f'ensemble_num: {i+1} acc: {acc:.6f} ece: {ece:.6f}')

if __name__ == '__main__':
    main()