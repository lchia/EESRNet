import yaml
import os
import os.path as osp
import logging
import sys
sys.path.append('../')
from utils import logger
import shutil
import tensorflow as tf

def parse(opt):
    path, name, resume = opt.opt, opt.name, opt.resume

    with open(path, 'r') as fp:
        args = yaml.full_load(fp.read())
    lg = logger(name, 'experiments/log/{}.log'.format(name), resume)

    # general settings
    args['name'] = name

    # dataset settings
    for phase, dataset_opt in args['datasets'].items():
        dataset_opt['scale'] = opt.scale
        dataset_opt['split'] = phase
        dataset_opt['patch_size'] = opt.ps
        dataset_opt['batch_size'] = opt.bs
        if 'ts' in opt:
            dataset_opt['temporal_size'] = opt.ts
            if phase == 'val'and not dataset_opt['first_k'] is None:
                dataset_opt['first_k'] = max(opt.ts, dataset_opt['first_k'])

        dataset_opt['dataroot_LR'] = dataset_opt['dataroot_LR'].replace('N', str(opt.scale))

    # network settings
    args['networks']['scale'] = opt.scale
    args['networks']['num_fea'] = opt.num_fea
    args['networks']['m'] = opt.m
    args['networks']['netname'] = opt.netname

    # create experiment root
    args['solver']['resume'] = resume
    args['solver']['qat'] = opt.qat
    root = osp.join(args['paths']['experiment_root'], name)
    args['paths']['root'] = root
    args['paths']['ckp'] = osp.join(root, 'best_status')
    args['paths']['visual'] = osp.join(root, 'visual')
    args['paths']['state'] = osp.join(root, 'state.pkl')


    args['solver']['finetune'] = opt.finetune
    args['solver']['pretrain_path'] = opt.pretrain_path

    if osp.exists(root) and resume==False:
        lg.info('Remove dir: [{}]'.format(root))
        shutil.rmtree(root, True)
    for name, path in args['paths'].items(): 
        if name == 'state':
            continue
        if not osp.exists(path):
            os.mkdir(path)
            lg.info('Create directory: {}'.format(path)) 
    
    # solver
    args['solver']['lr'] = opt.lr
    args['solver']['qat_path'] = opt.qat_path
    args['solver']['resume_path'] = opt.resume_path
    args['solver']['loss'] = opt.loss
    args['solver']['lr_policy'] = opt.lr_policy
    args['solver']['lr_minimum'] = opt.lr_minimum

    if opt.epochs == 1:
        args['solver']['epochs'] = opt.epochs

    # GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    lg.info('Available gpu: {}'.format(opt.gpu_ids))

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    lg.info('Set gpu: set_memory_growth=True')
    
    return dict_to_nonedict(args), lg


class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        for k,v in opt.items():
            opt[k] = dict_to_nonedict(v)
        return NoneDict(**opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(x) for x in opt]
    else:
        return opt
