from pathlib import Path
import myutils
import argparse
import torch
from torchvision import transforms
import datetime
from easydict import EasyDict as edict
import os, logging, sys

def get_config():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--use_dataset', type=str, default='Cars', choices=['CUB', 'Cars', 'SOP', 'Inshop'])
    # batch
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--instances', type=int, default=3)
    # optimization
    parser.add_argument('--lr', type=float, default=0.0)
    parser.add_argument('--lr_p', type=float, default=0.0)
    parser.add_argument('--lr_gamma', type=float, default=0.0)
    # model dataset
    parser.add_argument('--freeze_bn', type=int, default=1)
    # method
    parser.add_argument('--use_loss', type=str, default='triplet', choices=['triplet', 'n-npair', 'semihtriplet', 'ms'])
    parser.add_argument('--sec_wei', type=float, default=0.0)
    parser.add_argument('--norm_momentum', type=float, default=1.0)
    parser.add_argument('--l2reg_wei', type=float, default=0.0)

    parser.add_argument('--test_sop_model', type=str, default='')
    
    conf = parser.parse_args()

    conf.num_devs = 1

    if conf.use_dataset == 'CUB':
        conf.lr = 1.0e-5 if conf.lr==0 else conf.lr
        conf.lr_p = 0.5e-5 if conf.lr_p==0 else conf.lr_p
        conf.weight_decay = 0.5 * 5e-3

        conf.start_step = 0
        conf.lr_gamma = 0.1 if conf.lr_gamma==0 else conf.lr_gamma
        if conf.use_loss=='ms':
            conf.step_milestones = [3000, 6000, 9000]
        else:
            conf.step_milestones = [5000, 9000, 9000]
        conf.steps = 8000

    elif conf.use_dataset == 'Cars':
        conf.lr = 1e-5 if conf.lr==0 else conf.lr
        conf.lr_p = 1e-5 if conf.lr_p==0 else conf.lr_p
        conf.weight_decay = 0.5 * 5e-3

        conf.start_step = 0
        if conf.lr_gamma == 0.1:
            conf.step_milestones = [2000, 9000, 9000]
        elif conf.lr_gamma == 0.5:
            conf.step_milestones = [4000, 6000, 9000]
        conf.steps = 8000

    elif conf.use_dataset == 'SOP':
        conf.lr = 2.5e-4 if conf.lr==0 else conf.lr
        conf.lr_p = 0.5e-4 if conf.lr_p==0 else conf.lr_p
        conf.weight_decay = 1e-5
        
        conf.start_step = 0
        conf.lr_gamma = 0.1 if conf.lr_gamma==0 else conf.lr_gamma
        conf.step_milestones = [6e3, 18e3, 35e3]
        conf.steps = 12e3

    elif conf.use_dataset == 'Inshop':
        conf.lr = 5e-4 if conf.lr==0 else conf.lr
        conf.lr_p = 1e-4 if conf.lr_p==0 else conf.lr_p
        conf.weight_decay = 1e-5

        conf.start_step = 0
        conf.lr_gamma = 0.1 if conf.lr_gamma==0 else conf.lr_gamma
        conf.step_milestones = [6e3, 18e3, 35e3]
        conf.steps = 12e3

    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    now_time = datetime.datetime.now().strftime('%m%d_%H%M')
    conf_work_path = 'work_space/' + conf.use_dataset + '_' + now_time
    myutils.mkdir_p(conf_work_path, delete=True)
    myutils.set_file_logger(work_dir=conf_work_path, log_level=logging.DEBUG)
    sys.stdout = myutils.Logger(conf_work_path + '/log-prt')
    sys.stderr = myutils.Logger(conf_work_path + '/log-prt-err')

    path0, path1 = conf_work_path.split('/')
    conf.log_path = Path(path0) / 'logs' / path1 / 'log'
    conf.work_path = Path(conf_work_path)
    conf.model_path = conf.work_path / 'models'
    conf.save_path = conf.work_path / 'save'
    
    conf.start_eval = False

    conf.num_workers = 8

    conf.bninception_pretrained_model_path = './pretrained_models/bn_inception-52deb4733.pth'

    conf.transform_dict = {}
    conf.use_simple_aug = False

    conf.transform_dict['rand-crop'] = \
        transforms.Compose([
            transforms.Resize(size=(256, 256)) if conf.use_simple_aug else transforms.Resize(size=256),
            transforms.RandomCrop((227, 227)) if conf.use_simple_aug else transforms.RandomResizedCrop(
                                                                              scale=[0.16, 1],
                                                                              size=227
                                                                          ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123 / 255.0, 117 / 255.0, 104 / 255.0],
                                 std=[1.0 / 255, 1.0 / 255, 1.0 / 255]),
            transforms.Lambda(lambda x: x[[2, 1, 0], ...]) #to BGR
        ])
    conf.transform_dict['center-crop'] = \
        transforms.Compose([
            transforms.Resize(size=(256, 256)) if conf.use_simple_aug else transforms.Resize(size=256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123 / 255.0, 117 / 255.0, 104 / 255.0],
                                 std=[1.0 / 255, 1.0 / 255, 1.0 / 255]),
            transforms.Lambda(lambda x: x[[2, 1, 0], ...]) #to BGR
        ])
    


    return conf
