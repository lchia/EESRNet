## Mobile AI & AIM 2022 Real-Time Video Super-Resolution Challenge
* Team: NCUT-VGroup

* Method: <EESRNet: A Network for Energy Efficient Super Resolution>

* Submission file: **Mobile AI & AIM 2022 Real-Time VSR Challenge ysj_1.zip**

  * Firstly, <u>unzip</u> the submitted zip file and <u>assume</u> the directory is the **ROOT** directory.

    ```shell
    ROOT=Mobile AI & AIM 2022 Real-Time VSR Challenge ysj_1
    ```
  
  * Then <u>cd</u> to the `${ROOT}/Source-Codes` directory.

## 1. Environments
<u>Prepare</u> a conda environment to run our codes, for example, mai22.

```shell
conda create -y -n mai22 python=3.7
conda activate mai22
pip install tensorflow
pip install tensorflow-model-optimization
pip install tensorboardx
pip install opencv-python
pip install blessings
pip install imageio
pip install psutil
pip install scipy
pip install tb-nightly
pip install tf-estimator-nightly
pip install pyyaml
pip install matplotlib
pip install tqdm
conda install cudatoolkit=11.2 cudnn -c conda-forge
```




## 2. Dataset
<u>Prepare</u> REDS Training/Validation/Test set as follows.

* <u>Download</u> [REDS](https://codalab.lisn.upsaclay.fr/competitions/1756#participate) and put `${ROOT}/Source-Codes/data/REDS`, and generate meta Info files.

* Dataset structure

  > ${ROOT}/Source-Codes/data/REDS
  >
  > > train/train_sharp_bicubic/X4
  > >
  > > > 000
  > > >
  > > > > 00000000.png, 00000001.png, ...
  > > >
  > > > 001
  > > >
  > > > > 00000000.png, 00000001.png, ...
  > > >
  > > > ...
  >
  > > train/train_sharp
  >
  > > val/val_sharp_bicubic
  >
  > > val/val_sharp
  >
  > > test/test_sharp_bicubic

* Meta info files

    ```shell
    # generate meta_data
    python data/generate_meta_info.py --dataset reds
    ```

​		This <u>generates</u> `REDS_train.txt, REDS_val.txt` in folder `${ROOT}/Source-Codes/data`.


## 3. Test
<u>Reproduce</u> the final reconstructed frames (test set) as follows. The frames are uploaded to [google drive](https://drive.google.com/file/d/1eVoCSIkNhyhe12H640e1gDdu6FVH5DRs/view?usp=sharing).

```shell
python test_reds.py --input_shape 1,180,320,3 --gpu_ids 0
```

All the reconstructed frames lie in folder `${ROOT}/Source-Codes/final_reconstructed_frames`. 

**NOTE**: The  test script loads the submitted TFLite model `${ROOT}/TFLite/model.tflite`.




## 4. TFLite
Steps to restore our final model and convert it to TFLite.

* Our final model lies in `${ROOT}/Model/checkpoint`

* Steps to <u>generate</u> TFLite model from our final model.

  ```shell
  cd ${ROOT}/Model
  python main.py --gpu_ids 1 --input_shape 1,180,320,3
  ```
  
  * The newly generated TFLite model lies in `${ROOT}/Model/TFLite`, which is the same to `${ROOT}/TFLite/model.tflite`.


## 5. Train 
Pipeline to train our final model. 

### 5.1 Train *FP32* Model 

```shell
# FP32 
python train.py --opt options/train/reds_single_full.yaml --netname eesrnet --name reds_single_eesrnet --dataset reds --dataset_type single --m 0 --num_fea 16 --scale 4 --bs 16 --ps 64 --lr 1e-3 --lr_policy CosineAnnealingLR --gpu_ids 0
```

* The argument ```--name``` specifies the following save path:

* Log file will be saved in ```./experiments/log/reds_single_eesrnet.log```
* Checkpoint and current best weights will be saved in ```./experiments/training/reds_single_eesrnet/best_status/```
* Visualization of Train and Validate will be saved in ```./experiments/Tensorboard/reds_single_eesrnet/```



### 5.2 Train *INT8* Model

```shell
# INT8
python train.py --opt options/train/reds_single_full_qat.yaml --netname eesrnet --name reds_single_eesrnet_qat --dataset reds --dataset_type single --m 0 --num_fea 16 --scale 4 --bs 16 --ps 64 --lr 1e-3 --lr_policy CosineAnnealingLR --gpu_ids 0 --qat --qat_path experiments/training/reds_single_eesrnet/best_status
```

Our final model will be saved in `./experiments/training/reds_single_eesrnet_qat/best_status/`, which is same to the checkpoint files in `${ROOT}/Model/checkpoint`.

## Citation

Use this bibtex to cite this repository:
```
@InProceedings{ysj2022mai,
  title={EESRNet: A Network for Energy Efficient Super Resolution},
  author={Shijie Yue, etc.},
  year={2022},
  booktitle = {MAI22}
}
```

# Acknowledgement

Thanks to the organizers of the MAI&AIM 2022 Challenge for the perfect organization.

This implementations are inspired by following excellent project:

* [**SR_Mobile_Quantization**] (https://github.com/NJU-Jet/SR_Mobile_Quantization).

