import h5py
import numpy as np
import tensorflow as tf
import cv2
import random
import os
import os.path as osp
import pickle
from tqdm import tqdm 

class DIV2K(tf.keras.utils.Sequence):
    def __init__(self, opt):
        # convert .png file to .pt for faster loading
        self.opt = opt
        self.convert_img_to_pt(key='dataroot_HR')
        self.convert_img_to_pt(key='dataroot_LR')

        self.dataroot_hr = opt['dataroot_HR']
        self.dataroot_lr = opt['dataroot_LR']
        self.filename_path = opt['filename_path']
        self.scale = opt['scale']
        self.split = opt['split']
        self.patch_size = opt['patch_size']
        self.batch_size = opt['batch_size']
        self.flip = opt['flip']
        self.rot = opt['rot']
        self.enlarge_times = opt['enlarge_times']

        self.img_list = []
        with open(self.filename_path, 'r') as f:
            filenames = f.readlines()
        for line in filenames:
            self.img_list.append(line.strip())

    def convert_img_to_pt(self, key):
        if self.opt[key][-1] == '/':
            self.opt[key] = self.opt[key][:-1]
        img_list = os.listdir(self.opt[key])
        
        need_convert = False
        for i in range(len(img_list)):
            _, ext = osp.splitext(img_list[i])
            if ext != '.pt':
                need_convert = True
                break
        if need_convert == False:
            return
        
        new_dir_path = self.opt[key] + '_pt'
        if osp.exists(new_dir_path) and len(os.listdir(new_dir_path))==len(img_list):
            self.opt[key] = new_dir_path
            return

        os.makedirs(new_dir_path, exist_ok=True) 
        pbar = tqdm(total=len(frame_list))
        for i in range(len(frame_list)):
            base, ext = osp.splitext(frame_list[i])
            src_path = osp.join(self.opt[key], frame_list[i])
            dst_path = osp.join(new_dir_path, base+'.pt') 
            os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
            with open(dst_path, 'wb') as _f:
                img = cv2.imread(src_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pickle.dump(img, _f)
            pbar.set_description(f'>Convert to pt: {i+1}/{len(frame_list)}')
            pbar.update(1)
        self.opt[key] = new_dir_path

    def shuffle(self):
        random.shuffle(self.img_list)

    def __len__(self):
        if self.split == 'train':
            return int(len(self.img_list)*self.enlarge_times/self.batch_size)
            
        else:
            return len(self.img_list)

    def __getitem__(self, idx):
        start = (idx*self.batch_size) 
        end = start + self.batch_size
        if self.split == 'train':
            lr_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype=np.float32)
            hr_batch = np.zeros((self.batch_size, self.patch_size*self.scale, self.patch_size*self.scale, 3), dtype=np.float32)
            for i in range(start, end):
                lr, hr = self.get_image_pair(i%len(self.img_list))
                lr_batch[i-start] = lr
                hr_batch[i-start] = hr
        else:
            lr, hr = self.get_image_pair(idx)
            lr_batch, hr_batch = np.expand_dims(lr, 0), np.expand_dims(hr, 0)

        return (lr_batch).astype(np.float32), (hr_batch).astype(np.float32)

    def get_image_pair(self, idx):
        hr_path = osp.join(self.dataroot_hr, self.img_list[idx])
        base, ext = osp.splitext(self.img_list[idx])
        lr_basename = base + 'x{}'.format(self.scale) + '.pt'
        lr_path = osp.join(self.dataroot_lr, lr_basename)
        
        # load img
        hr = self.read_img(hr_path)
        lr = self.read_img(lr_path)

        if self.split == 'train':
            lr_patch, hr_patch = self.get_patch(lr, hr, self.patch_size, self.scale)
            lr, hr = self.augment(lr_patch, hr_patch, self.flip, self.rot)
        
        return lr, hr

    def read_img(self, img_path):
        with open(img_path, 'rb') as f:
            img = pickle.load(f)

        return img

    def get_patch(self, lr, hr, ps, scale):
        lr_h, lr_w = lr.shape[:2]
        hr_h, hr_w = hr.shape[:2]

        lr_x = random.randint(0, lr_w - ps)
        lr_y = random.randint(0, lr_h - ps)
        hr_x = lr_x * scale
        hr_y = lr_y * scale

        lr_patch = lr[lr_y:lr_y+ps, lr_x:lr_x+ps, :]
        hr_patch = hr[hr_y:hr_y+ps*scale, hr_x:hr_x+ps*scale, :]

        return lr_patch, hr_patch

    def augment(self, lr, hr, flip, rot):
        hflip = flip and random.random() < 0.5
        vflip = flip and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip:
            lr = np.ascontiguousarray(lr[:, ::-1, :])
            hr = np.ascontiguousarray(hr[:, ::-1, :])
        if vflip:
            lr = np.ascontiguousarray(lr[::-1, :, :])
            hr = np.ascontiguousarray(hr[::-1, :, :])
        if rot90:
            lr = lr.transpose(1, 0, 2)
            hr = hr.transpose(1, 0, 2)
    
        return lr, hr


class REDS(tf.keras.utils.Sequence):
    def __init__(self, opt):
        # convert .png file to .pt for faster loading 
        self.opt = opt
        self.convert_img_to_pt(key='dataroot_HR')
        self.convert_img_to_pt(key='dataroot_LR')

        self.dataroot_hr = opt['dataroot_HR']
        self.dataroot_lr = opt['dataroot_LR']
        self.filename_path = opt['filename_path']
        self.scale = opt['scale']
        self.split = opt['split']
        self.patch_size = opt['patch_size']
        self.batch_size = opt['batch_size']
        self.temporal_size = opt['temporal_size']
        self.temporal_type = opt['temporal_type']

        self.flip = opt['flip']
        self.rot = opt['rot']
        self.enlarge_times = opt['enlarge_times']
        self.filename_tmpl = opt['filename_tmpl']
        self.filename_ext = opt['filename_ext']
        self.combine_channel_temporal = opt['combine_channel_temporal']
        self.first_k = opt['first_k']

        # print(f'\t self.dataroot_hr: {self.dataroot_hr}')
        # print(f'\t self.dataroot_lr: {self.dataroot_lr}')
        # print(f'\t self.filename_ext: {self.filename_ext}')
 

        self.img_list = []
        self.total_num_frames = [] # some clips may not have 100 frames
        self.start_frames= [] # some clips may not start from 00000
        with open(self.filename_path, 'r') as f:
            filenames = f.readlines()
        for line in filenames:
            folder, frame_num, _, start_frame = line.split(' ')

            this_frames = [f'{folder}/{i:{self.filename_tmpl}}' for i in range(int(start_frame)+self.temporal_size//2, int(start_frame)+int(frame_num), self.temporal_size)]
            
            if not self.first_k is None:
                this_frames = this_frames[:self.first_k]

            self.img_list.extend(this_frames)
            self.total_num_frames.extend([int(frame_num) for i in range(len(this_frames))])
            self.start_frames.extend([int(start_frame) for i in range(len(this_frames))])

    def get_all_video_frames(self, video_root): 
        if not os.path.exists(video_root):
            frame_list = []
        else: 
            video_list = os.listdir(video_root)
            frame_list = []
            for video_dir in video_list:
                video_path = os.path.join(video_root, video_dir)
                if not os.path.isdir(video_path):
                    continue
                else:
                    video_frames = os.listdir(video_path)
                    frame_list.extend([os.path.join(video_dir, frame_name) for frame_name in video_frames])

        return frame_list
 
    def convert_img_to_pt(self, key): 
        if self.opt[key][-1] == '/':
            self.opt[key] = self.opt[key][:-1]
        frame_list = self.get_all_video_frames(self.opt[key])

        need_convert = False
        for i in range(len(frame_list)):
            _, ext = osp.splitext(frame_list[i])
            if ext != '.pt':
                need_convert = True
                break
        if need_convert == False:
            return
        
        new_dir_path = self.opt[key] + '_pt'
        new_frame_list = self.get_all_video_frames(new_dir_path)
        if osp.exists(new_dir_path) and len(new_frame_list)==len(frame_list):
            self.opt[key] = new_dir_path
            self.opt['filename_ext'] = 'pt'
            return

        os.makedirs(new_dir_path, exist_ok=True)
        pbar = tqdm(total=len(frame_list))
        for i in range(len(frame_list)):
            base, ext = osp.splitext(frame_list[i])
            src_path = osp.join(self.opt[key], frame_list[i])
            dst_path = osp.join(new_dir_path, base+'.pt') 
            os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
            with open(dst_path, 'wb') as _f:
                img = cv2.imread(src_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pickle.dump(img, _f)
            pbar.set_description(f'>Convert to pt: {i+1}/{len(frame_list)}')
            pbar.update(1)
        self.opt[key] = new_dir_path
        self.opt['filename_ext'] = 'pt'

    def shuffle(self):
        random.shuffle(self.img_list)

    def __len__(self):
        if self.split == 'train':
            return int(len(self.img_list)*self.enlarge_times/self.batch_size)
            
        else:
            return len(self.img_list)

    def __getitem__(self, idx):
        start = (idx*self.batch_size)
        end = start + self.batch_size
        if self.split == 'train':
            if not self.combine_channel_temporal:
                lr_batch = np.zeros((self.batch_size, self.temporal_size, self.patch_size, self.patch_size, 3), dtype=np.float32)
                hr_batch = np.zeros((self.batch_size, self.temporal_size, self.patch_size*self.scale, self.patch_size*self.scale, 3), dtype=np.float32)
            else:
                lr_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3*self.temporal_size), dtype=np.float32)
                hr_batch = np.zeros((self.batch_size, self.patch_size*self.scale, self.patch_size*self.scale, 3*self.temporal_size), dtype=np.float32)
            
            for i in range(start, end):
                lr, hr = self.get_image_pair(i%len(self.img_list))
                lr_batch[i-start] = lr
                hr_batch[i-start] = hr
        else:
            lr, hr = self.get_image_pair(idx)
            lr_batch, hr_batch = np.expand_dims(lr, 0), np.expand_dims(hr, 0)

        return (lr_batch).astype(np.float32), (hr_batch).astype(np.float32)

    def get_sequence(self, center_frame_idx, temporal_size, video_start_frame, video_frame_num):
        """Get filenames of sequence of frames.

        Args:
            center_frame_idx:
            temporal_size:
            video_start_frame:
            video_frame_num:

        Returns:
            sequence: A `list` contains filenames of sequence of frames. 
        """
        sequence = []
        if temporal_size % 2 == 0:
            num_frames_left  = temporal_size // 2 - 1
            num_frames_right = temporal_size // 2
            clip_start_idx = center_frame_idx - num_frames_left
            clip_stop_idx  = center_frame_idx + num_frames_right
        else:
            num_frames_side = temporal_size // 2
            clip_start_idx = center_frame_idx - num_frames_side
            clip_stop_idx  = center_frame_idx + num_frames_side
            num_frames_left  = num_frames_side
            num_frames_right = num_frames_side

        # frames - left
        idx_left = []
        for k in range(num_frames_left):
            idx_left.append(max(video_start_frame, center_frame_idx - (k+1)))

        idx_left = [idx_left[k] for k in list(range(len(idx_left)-1, -1, -1))]
        sequence.extend(idx_left)

        # center
        sequence.append(center_frame_idx)

        # frames - right
        for k in range(num_frames_right):
            sequence.append(min(video_frame_num-1, center_frame_idx + (k+1)))
 
        return sequence
 
    def get_image_pair(self, idx):
        video_frame_num = self.total_num_frames[idx]
        video_start_frame= self.start_frames[idx]
        clip_name, frame_name = self.img_list[idx].split('/')  # key example: 000/00000000 

        center_frame_idx = int(frame_name)
        neighbor_list = self.get_sequence(center_frame_idx, self.temporal_size, video_start_frame, video_frame_num)

        # print(f'\n************************************\n************************************')
        # print(f'idx: {idx}, temporal_size: {self.temporal_size}')
        # print(f'clip_name/frame_name: {self.img_list[idx]}')
        # print(f'neighbor: {neighbor_list}')

        hr_clip = []
        lr_clip = []
        for frame_idx in neighbor_list:
            # load img
            hr_path = osp.join(self.dataroot_hr, clip_name, f'{frame_idx:{self.filename_tmpl}}.{self.filename_ext}')
            lr_path = osp.join(self.dataroot_lr, clip_name, f'{frame_idx:{self.filename_tmpl}}.{self.filename_ext}') 

            hr = self.read_img(hr_path)
            lr = self.read_img(lr_path)

            if self.split == 'train':
                lr_patch, hr_patch = self.get_patch(lr, hr, self.patch_size, self.scale)
                lr, hr = self.augment(lr_patch, hr_patch, self.flip, self.rot) 

            # clip
            hr_clip.append(np.expand_dims(hr, axis=0))
            lr_clip.append(np.expand_dims(lr, axis=0))

        # concat
        hr = tf.concat(hr_clip, axis=0)
        lr = tf.concat(lr_clip, axis=0)
        # print(f'hr: {hr.shape}')
        # print(f'lr: {lr.shape}')

        if self.combine_channel_temporal:
            hr = tf.transpose(hr, (1, 2, 0, 3))
            lr = tf.transpose(lr, (1, 2, 0, 3))
            # print(f'hr: {hr.shape}')
            # print(f'lr: {lr.shape}')

            hr = tf.reshape(hr, (hr.shape[0], hr.shape[1], -1))
            lr = tf.reshape(lr, (lr.shape[0], lr.shape[1], -1))
            # print(f'hr: {hr.shape}')
            # print(f'lr: {lr.shape}')

        return lr, hr

    def read_img(self, img_path):
        with open(img_path, 'rb') as f:
            img = pickle.load(f)

        return img

    def get_patch(self, lr, hr, ps, scale):
        lr_h, lr_w = lr.shape[:2]
        hr_h, hr_w = hr.shape[:2]

        lr_x = random.randint(0, lr_w - ps)
        lr_y = random.randint(0, lr_h - ps)
        hr_x = lr_x * scale
        hr_y = lr_y * scale

        lr_patch = lr[lr_y:lr_y+ps, lr_x:lr_x+ps, :]
        hr_patch = hr[hr_y:hr_y+ps*scale, hr_x:hr_x+ps*scale, :]

        return lr_patch, hr_patch

    def augment(self, lr, hr, flip, rot):
        hflip = flip and random.random() < 0.5
        vflip = flip and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip:
            lr = np.ascontiguousarray(lr[:, ::-1, :])
            hr = np.ascontiguousarray(hr[:, ::-1, :])
        if vflip:
            lr = np.ascontiguousarray(lr[::-1, :, :])
            hr = np.ascontiguousarray(hr[::-1, :, :])
        if rot90:
            lr = lr.transpose(1, 0, 2)
            hr = hr.transpose(1, 0, 2)
    
        return lr, hr


class REDSImages(tf.keras.utils.Sequence):
    def __init__(self, opt):
        # convert .png file to .pt for faster loading 
        self.opt = opt
        self.convert_img_to_pt(key='dataroot_HR')
        self.convert_img_to_pt(key='dataroot_LR')

        self.dataroot_hr = opt['dataroot_HR']
        self.dataroot_lr = opt['dataroot_LR']
        self.filename_path = opt['filename_path']
        self.scale = opt['scale']
        self.split = opt['split']
        self.patch_size = opt['patch_size']
        self.batch_size = opt['batch_size']
        self.temporal_size = opt['temporal_size']
        self.temporal_type = opt['temporal_type']

        self.flip = opt['flip']
        self.rot = opt['rot']
        self.enlarge_times = opt['enlarge_times']
        self.filename_tmpl = opt['filename_tmpl']
        self.filename_ext = opt['filename_ext']
        self.combine_channel_temporal = opt['combine_channel_temporal']
        self.first_k = opt['first_k']

        print(f'\t self.dataroot_hr: {self.dataroot_hr}')
        print(f'\t self.dataroot_lr: {self.dataroot_lr}')
        print(f'\t self.filename_ext: {self.filename_ext}')
 

        self.img_list = []
        self.total_num_frames = [] # some clips may not have 100 frames
        self.start_frames= [] # some clips may not start from 00000
        with open(self.filename_path, 'r') as f:
            filenames = f.readlines()
        for line in filenames:
            folder, frame_num, _, start_frame = line.split(' ')

            this_frames = [f'{folder}/{i:{self.filename_tmpl}}' for i in range(int(start_frame), int(start_frame)+int(frame_num))]
            
            if not self.first_k is None:
                this_frames = this_frames[:self.first_k]

            self.img_list.extend(this_frames)

    def get_all_video_frames(self, video_root): 
        if not os.path.exists(video_root):
            frame_list = []
        else: 
            video_list = os.listdir(video_root)
            frame_list = []
            for video_dir in video_list:
                video_path = os.path.join(video_root, video_dir)
                if not os.path.isdir(video_path):
                    continue
                else:
                    video_frames = os.listdir(video_path)
                    frame_list.extend([os.path.join(video_dir, frame_name) for frame_name in video_frames])

        return frame_list
 
    def convert_img_to_pt(self, key): 
        if self.opt[key][-1] == '/':
            self.opt[key] = self.opt[key][:-1]
        frame_list = self.get_all_video_frames(self.opt[key])

        need_convert = False
        for i in range(len(frame_list)):
            _, ext = osp.splitext(frame_list[i])
            if ext != '.pt':
                need_convert = True
                break
        if need_convert == False:
            return
        
        new_dir_path = self.opt[key] + '_pt'
        new_frame_list = self.get_all_video_frames(new_dir_path)
        if osp.exists(new_dir_path) and len(new_frame_list)==len(frame_list):
            self.opt[key] = new_dir_path
            self.opt['filename_ext'] = 'pt'
            return

        os.makedirs(new_dir_path, exist_ok=True)
        pbar = tqdm(total=len(frame_list))
        for i in range(len(frame_list)):
            base, ext = osp.splitext(frame_list[i])
            src_path = osp.join(self.opt[key], frame_list[i])
            dst_path = osp.join(new_dir_path, base+'.pt') 
            os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
            with open(dst_path, 'wb') as _f:
                img = cv2.imread(src_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pickle.dump(img, _f)
            pbar.set_description(f'>Convert to pt: {i+1}/{len(frame_list)}')
            pbar.update(1)

        self.opt[key] = new_dir_path
        self.opt['filename_ext'] = 'pt'

    def shuffle(self):
        random.shuffle(self.img_list)

    def __len__(self):
        if self.split == 'train':
            return int(len(self.img_list)*self.enlarge_times/self.batch_size)
            
        else:
            return len(self.img_list)

    def __getitem__(self, idx):
        start = (idx*self.batch_size)
        end = start + self.batch_size
        if self.split == 'train': 
            lr_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype=np.float32)
            hr_batch = np.zeros((self.batch_size, self.patch_size*self.scale, self.patch_size*self.scale, 3), dtype=np.float32)
            for i in range(start, end):
                lr, hr = self.get_image_pair(i%len(self.img_list))
                lr_batch[i-start] = lr
                hr_batch[i-start] = hr
        else:
            lr, hr = self.get_image_pair(idx)
            lr_batch, hr_batch = np.expand_dims(lr, 0), np.expand_dims(hr, 0)

        return (lr_batch).astype(np.float32), (hr_batch).astype(np.float32)
 
    def get_image_pair(self, idx):
        image_key = self.img_list[idx]
  
        # load img
        hr_path = osp.join(self.dataroot_hr, f'{image_key}.{self.filename_ext}')
        lr_path = osp.join(self.dataroot_lr, f'{image_key}.{self.filename_ext}') 

        hr = self.read_img(hr_path)
        lr = self.read_img(lr_path)

        if self.split == 'train':
            lr_patch, hr_patch = self.get_patch(lr, hr, self.patch_size, self.scale)
            lr, hr = self.augment(lr_patch, hr_patch, self.flip, self.rot)

        return lr, hr

    def read_img(self, img_path):
        with open(img_path, 'rb') as f:
            img = pickle.load(f)

        return img

    def get_patch(self, lr, hr, ps, scale):
        lr_h, lr_w = lr.shape[:2]
        hr_h, hr_w = hr.shape[:2]

        lr_x = random.randint(0, lr_w - ps)
        lr_y = random.randint(0, lr_h - ps)
        hr_x = lr_x * scale
        hr_y = lr_y * scale

        lr_patch = lr[lr_y:lr_y+ps, lr_x:lr_x+ps, :]
        hr_patch = hr[hr_y:hr_y+ps*scale, hr_x:hr_x+ps*scale, :]

        return lr_patch, hr_patch

    def augment(self, lr, hr, flip, rot):
        hflip = flip and random.random() < 0.5
        vflip = flip and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip:
            lr = np.ascontiguousarray(lr[:, ::-1, :])
            hr = np.ascontiguousarray(hr[:, ::-1, :])
        if vflip:
            lr = np.ascontiguousarray(lr[::-1, :, :])
            hr = np.ascontiguousarray(hr[::-1, :, :])
        if rot90:
            lr = lr.transpose(1, 0, 2)
            hr = hr.transpose(1, 0, 2)
    
        return lr, hr

