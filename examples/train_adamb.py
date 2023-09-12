import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from typing import Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

### original ###
from optim import AdamB
from utils import set_seed, calc_acc, test, setup_dateset, get_model, save_params_dir
from args import parse

def main():

    hparams = parse()

    if hparams.is_debug:
        #### deterministic ###
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    set_seed(seed=hparams.seed, device=device) 

    model = get_model(hparams.model)
    model.to(device)    
    print(f'model: {hparams.model}')


    criterion = torch.nn.CrossEntropyLoss()

    ### setup AdamB ###
    opt_params = {
        "reg_factor": 1e-2,  # in case vanilla BBB, set N / batch_size value,
        "lr": 1e-3,
        "lr_rho": 1e-3,
        "betas": (0.9, 0.999),
        "init_std_scale": 1.0,
        "device": device,
        "log_s0": -1.0,
        "log_s1": -6.0,
    }
    optimizer = AdamB(model, **opt_params)

    if hparams.use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.epoch)

    trainloader, testloader, train_batch_size, test_batch_size = setup_dateset(hparams)

    sum_time = 0.0
    start_loop = time.time()
    for epoch in range(hparams.epoch):
        start = time.time()
        model.train()
        for i, data in enumerate(trainloader):
            x, t = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)

            def closure() -> Tuple[torch.Tensor, torch.Tensor]:
                optimizer.zero_grad()
                y = model(x)
                loss = criterion(y, t)
                correct_ans = calc_acc(y, t)
                acc = (correct_ans / train_batch_size) * 100
                loss.backward()
                return loss, acc

            loss, acc = optimizer.step(
                closure, sampling=True)

            if hparams.is_debug:
                print(f'{epoch + 1} epoch, {i + 1:5d} iter loss: {loss.item():.6f} acc: {acc:.6f}')         
                if i == 3:
                    break

        process_time = time.time() - start
        sum_time += process_time

        if hparams.use_scheduler:
            scheduler.step()

        if (epoch+1) % 10 == 0:
            test_loss, test_acc = test(model, testloader, criterion, device, optimizer)
            print(f'epoch: {epoch+1} test_loss: {test_loss:.6f} test_acc: {test_acc:.6f}')
        
    print('Finished Training')
    print(f'average 1 epoch training time: {sum_time / hparams.epoch:.6f}')
    print(f'total training time: {time.time() - start_loop:.6f}')

    save_dir = save_params_dir(hparams.save_dir, hparams.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f'save_dir: {save_dir}')
    torch.save(model.state_dict(),  os.path.join(save_dir, 'model.pth'))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pth'))

if __name__ == '__main__':
    main()