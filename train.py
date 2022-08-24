import os
import argparse
import cv2
import numpy as np
from options import parse
from solvers import Solver
from data import REDSImages, REDS
import matplotlib as mpl
import matplotlib.pyplot as plt
import shutil
import os
import os.path as osp
from tensorboardX import SummaryWriter
import tensorflow as tf

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FSRCNN Demo')
    parser.add_argument('--opt', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--scale', default=3, type=int)
    parser.add_argument('--num_fea', default=28, type=int, help='number of feature channels')
    parser.add_argument('--m', default=4, type=int, help='number of feature blocks') 
    parser.add_argument('--ps', default=48, type=int, help='patch_size')
    parser.add_argument('--bs', default=16, type=int, help='batch_size')
    parser.add_argument('--ts', default=7, type=int, help='temporal size')
    parser.add_argument('--loss', default='mae', type=str, help='train loss')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--lr_minimum', default=1e-8, type=float, help='learning rate minimum')
    parser.add_argument('--lr_policy', default='STEP', type=str, help='learning rate policy')
    parser.add_argument('--gpu_ids', default=None)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path', default=None)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--pretrain_path', default=None)
    parser.add_argument('--qat', action='store_true', default=False)
    parser.add_argument('--qat_path', default=None)
    parser.add_argument('--dataset', default='div2k', type=str, help='which dataset')
    parser.add_argument('--dataset_type', default='single', type=str, help='single, multiple')
    parser.add_argument('--netname', default='net1', type=str, help='select different network under a same name')
    parser.add_argument('--epochs', default=300, type=int, help='number of training epochs')

    args = parser.parse_args()

    # experiments
    os.makedirs('./experiments', exist_ok=True)
 
    dataset = args.dataset
    dataset_type = args.dataset_type
    netname = args.netname

    # assign new values to the variables in the config.yaml using the 'parser' value.
    args, lg = parse(args)

    lg.info('args: [{}]'.format(args))


    # Tensorboard save directory
    resume = args['solver']['resume']
    tensorboard_path = 'experiments/Tensorboard/{}'.format(args['name'])

    if resume==False:
        if osp.exists(tensorboard_path):
            shutil.rmtree(tensorboard_path, True)
            lg.info('Remove dir: [{}]'.format(tensorboard_path))
    writer = SummaryWriter(tensorboard_path)

    # create dataset
    if dataset == 'div2k':
        # DIV2k

        train_data = DIV2K(args['datasets']['train'])
        lg.info('Create train dataset successfully!')
        lg.info('Training: [{}] iterations for each epoch'.format(len(train_data)))

        val_data = DIV2K(args['datasets']['val'])
        lg.info('Create val dataset successfully!')
        lg.info('Validating: [{}] iterations for each epoch'.format(len(val_data)))

        # adding name of network: {which_model}_{network}
        args['networks']['netname'] = netname

        # create solver
        lg.info('Preparing for experiment: [{}]'.format(args['name']))
        solver = Solver(args, train_data, val_data, writer)

        # train
        lg.info('Start training...')
        solver.train()

    elif dataset == 'imagenet':
        # DIV2k

        print(f'\n\t config train > ', args['datasets']['train'])

        train_data = ImageNet(args['datasets']['train'])
        lg.info('Create train dataset successfully!')
        lg.info('Training: [{}] iterations for each epoch'.format(len(train_data)))

        val_data = REDSImages(args['datasets']['val'])
        lg.info('Create val dataset successfully!')
        lg.info('Validating: [{}] iterations for each epoch'.format(len(val_data)))

        # adding name of network: {which_model}_{network}
        args['networks']['netname'] = netname

        # create solver
        lg.info('Preparing for experiment: [{}]'.format(args['name']))
        solver = Solver(args, train_data, val_data, writer)

        # train
        lg.info('Start training...')
        solver.train()

    elif dataset == 'reds':
        # REDS

        if dataset_type == 'single': 
            train_data = REDSImages(args['datasets']['train']) 
            lg.info('Create train dataset successfully!')
            lg.info('Training: [{}] iterations for each epoch'.format(len(train_data)))

            val_data = REDSImages(args['datasets']['val'])        
            lg.info('Create val dataset successfully!')
            lg.info('Validating: [{}] iterations for each epoch'.format(len(val_data)))
            
            # adding name of network: {which_model}_{network}
            args['networks']['netname'] = netname

            # create solver
            lg.info('Preparing for experiment: [{}]'.format(args['name']))
            solver = Solver(args, train_data, val_data, writer)

            # train
            lg.info('Start training...')
            solver.train()
 
        elif dataset_type == 'multiple':
            train_data = REDS(args['datasets']['train']) 
            lg.info('Create train dataset successfully!')
            lg.info('Training: [{}] iterations for each epoch'.format(len(train_data)))
            
            val_data = REDS(args['datasets']['val'])     
            lg.info('Create val dataset successfully!')
            lg.info('Validating: [{}] iterations for each epoch'.format(len(val_data)))
            
            # adding name of network: {which_model}_{network}
            args['networks']['netname'] = netname

            # create solver
            lg.info('Preparing for experiment: [{}]'.format(args['name']))
            solver = Solver(args, train_data, val_data, writer)

            # train
            lg.info('Start training...')
            solver.train()

        else:
            lg.info('dataset_type: select single or multiple, only this two!')

    else:
        lg.info('dataset: select div2k, reds, now only support this two dataset!')
