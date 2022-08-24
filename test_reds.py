import os
import cv2
import pickle
import argparse
import numpy as np
import os.path as osp 
import tensorflow as tf
from tqdm import tqdm 

def reds_int8_test(fp32_model_path, save_path):

    interpreter = tf.lite.Interpreter(model_path=fp32_model_path)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    pbar = tqdm(total=30*100)
    for i in range(0, 30):
        video_key = f'{i:03d}'

        for j in range(0, 100):
            frame_key = f'{j:08d}'

            lr_path = f'./data/REDS/test/test_sharp_bicubic/X4_pt/{video_key}/{frame_key}.pt'
            with open(lr_path, 'rb') as f:
                lr = pickle.load(f)
            h, w, c = lr.shape
            lr = np.expand_dims(lr, 0).astype(np.uint8)
            #lr = np.round(lr/IS+IZ).astype(np.uint8)
            lr = lr.astype(np.uint8)


            interpreter.resize_tensor_input(input_details[0]['index'], lr.shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_details[0]['index'], lr)
            interpreter.invoke()

            sr = interpreter.get_tensor(output_details[0]['index'])

            sr = np.clip(sr, 0, 255)
            b, h, w, c = sr.shape
            # save image
            save_name = osp.join(save_path, f'{video_key}/{frame_key}.png')
            os.makedirs(osp.split(save_name)[0], exist_ok=True)
            cv2.imwrite(save_name, cv2.cvtColor(sr.squeeze().astype(np.float32), cv2.COLOR_RGB2BGR))

            pbar.set_description(f'>Test REDS: {i+1}/{30}, {j+1}/{100}')
            pbar.update(1)

    print('>test done.')
 
def get_all_video_frames(video_root): 
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

def convert_img_to_pt(dataroot_LR):

    frame_list = get_all_video_frames(dataroot_LR)

    need_convert = False
    for i in range(len(frame_list)):
        _, ext = osp.splitext(frame_list[i])
        if ext != '.pt':
            need_convert = True
            break
    if need_convert == False:
        return
    
    new_dir_path = dataroot_LR + '_pt'
    new_frame_list = get_all_video_frames(new_dir_path)
    if osp.exists(new_dir_path) and len(new_frame_list)==len(frame_list):
        dataroot_LR = new_dir_path
        filename_ext = 'pt'
        return

    os.makedirs(new_dir_path, exist_ok=True)
    pbar = tqdm(total=len(frame_list))
    for i in range(len(frame_list)):
        base, ext = osp.splitext(frame_list[i])
        src_path = osp.join(dataroot_LR, frame_list[i])
        dst_path = osp.join(new_dir_path, base+'.pt') 
        os.makedirs(os.path.split(dst_path)[0], exist_ok=True)
        with open(dst_path, 'wb') as _f:
            img = cv2.imread(src_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pickle.dump(img, _f)
        pbar.set_description(f'>Convert to pt: {i+1}/{len(frame_list)}')
        pbar.update(1)

if __name__ == '__main__':
    print('Start converting img to pt ......')
    dataroot_LR='./data/REDS/test/test_sharp_bicubic/X4'
    convert_img_to_pt(dataroot_LR)
    print('Successfully converted img to pt')

    parser = argparse.ArgumentParser() 
    parser.add_argument(
        '--input_shape', type=str, help='input shape.') 
    parser.add_argument(
        '--gpu_ids', default='0', type=str, help='0')  

    args = parser.parse_args()
   
    input_shape = [int(s) for s in args.input_shape.replace(' ', '').split(',')]

    gpu_ids = args.gpu_ids
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
  
    model_path = '../TFLite/model.tflite'
    save_path = './final_reconstructed_frames/'
    os.makedirs(save_path, exist_ok=True)
 
    reds_int8_test(model_path, save_path) 

