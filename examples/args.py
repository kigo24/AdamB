import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_debug", default=False, type=bool)
    parser.add_argument("--save_dir", default="save_params", type=str)

    parser.add_argument("--model", default="resnet18", type=str)

    """ Training Conditions """
    parser.add_argument("--epoch", default=210, type=int)
    parser.add_argument("--use_scheduler", default=True, type=bool)
    parser.add_argument("--seed", default=100, type=int)    
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--lr_rho", default=0.001, type=float)
    parser.add_argument("--init_std_scale", default=1.0, type=float)

    """ Inference Conditions """
    parser.add_argument("--ensemble_num", default=20, type=int)
    parser.add_argument("--inference_seed", default=1234, type=int) 

    args = parser.parse_args()
    return args