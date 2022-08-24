import argparse
from os import path as osp
from PIL import Image
 
import os
import cv2
import numpy as np
import random
 

def generate_meta_info_reds():
    """Generate meta info for SRQE2022 dataset, &
       randomly split a validation part.

    type2: list all sequences and the num of the frames of each sequece, like
        000 100 (720,1280,3) 00000000
        001 100 (720,1280,3) 00000000
        002 100 (720,1280,3) 00000000
        003 100 (720,1280,3) 00000000
        ...
    """
    set_names = {'train', 'val'}

    for setname in set_names:

        dataroot = f'./data/REDS/{setname}_sharp'
        meta_info_txt = f'./data/REDS_{setname}.txt'
           
        with open(meta_info_txt, 'w') as f:
            video_folders = sorted(os.listdir(dataroot)) 

            # each video
            idx = 0
            for video_folder in video_folders:
                frames = sorted(os.listdir(os.path.join(dataroot, video_folder)))

                num_frames = len(frames) 
                img_fullpath = os.path.join(dataroot, video_folder, frames[0])
                img = Image.open(img_fullpath)

                width, height = img.size
                mode = img.mode
                if mode == 'RGB':
                    n_channel = 3
                elif mode == 'L':
                    n_channel = 1
                else:
                    raise ValueError(f'Unsupported mode {mode}.')

                start_frame = os.path.splitext(frames[0])[0]

                info = f'{video_folder} {num_frames} ({height},{width},{n_channel}) {start_frame}'
                print('\t{} : {}'.format(idx + 1, info))
                f.write(f'{info}\n')
                idx += 1


def generate_meta_info_imagenet():  
    dataroot = f'./data/ImageNet/gt_pt'
    meta_info_txt = f'./data/ImageNet.txt'
    
    print('begin ...')
    with open(meta_info_txt, 'w') as f:
        img_list = sorted(os.listdir(dataroot)) 
 
        for img in img_list: 
            f.write(f'{img}\n')

            img_lr = f'./data/ImageNet/lr_pt/{img}'
            assert osp.exists(img_lr), f'file {img_lr} not exist.'

    print('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        help=(
            "Options: 'REDS' "
            'You may need to modify the corresponding configurations in codes.'
        ))

    args = parser.parse_args()
    dataset = args.dataset.lower()


    if dataset == 'reds':
        generate_meta_info_reds()
    elif dataset == 'imagenet':
        generate_meta_info_imagenet()