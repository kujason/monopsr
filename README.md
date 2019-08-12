# MonoPSR (CVPR 2019)
[Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction](https://arxiv.org/abs/1904.01690)

[Jason Ku*](http://kujason.com/), [Alex D. Pon*](http://alexdpon.com/), [Steven L. Waslander](https://scholar.google.ca/citations?user=CwgGTXMAAAAJ) (*Equal Contribution)

This repository contains the public release of the Tensorflow implementation of *Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction*.

## Video
Demo [video](https://youtu.be/_iJpEpXB7j4) showing results on several KITTI sequences.

## Getting Started
Implemented and tested on Ubuntu 16.04 with Python 3.5 and Tensorflow 1.8.0.

Clone this repo
```
git clone git@github.com:kujason/monopsr.git
```
Install Python dependencies
```
cd monopsr
pip3 install -r requirements.txt
```
Add monopsr/src to your PYTHONPATH
```
# For virtualenvwrapper users
add2virtualenv src/.
```

Compile the two custom TF ops `src/tf_ops/nn_distance` and `src/tf_ops/approxmatch` by running the shell scripts found in the respective folders. The location of your TensorFlow python package is passed as an argument.

For example:
```
sh src/tf_ops/approxmatch/tf_approxmatch_compile.sh ${HOME}/.virtualenvs/{monopsr}/lib/python3.5/site-packages/tensorflow
```
```
sh src/tf_ops/nn_distance/tf_nndistance_compile.sh ${HOME}/.virtualenvs/{monopsr}/lib/python3.5/site-packages/tensorflow
```

## Training
To train on the [KITTI Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):

Download the data and place it in your home folder at `~/Kitti/object`.

Go [here](https://drive.google.com/open?id=17dUeXJWSFTxmbmbXC0--VqUXWenkceEb) and download the train.txt, val.txt and trainval.txt splits into `~/Kitti/object`.
```
/home/$USER/Kitti
    object
        testing
        training
            calib
            image_2
            label_2
            velodyne
        train.txt
        trainval.txt
        val.txt
```

### 2D Detections
Download the MSCNN 2D detections [here](https://drive.google.com/open?id=17dUeXJWSFTxmbmbXC0--VqUXWenkceEb) and place it in 
`monopsr/data/detections/mscnn`

### Depth Maps and Instance Masks
Generate the ground truth depth maps and instance segmentation:
```
python demos/depth_completion/save_lidar_depth_maps.py
python demos/instances/gen_instance_masks.py
```
Place the depth maps and segmentation outputs in `~/Kitti/object/training/`.
```
/home/$USER/Kitti
    object
        testing
        training
            calib
            *depth_2_multiscale
            image_2
            *instance_2_multiscale
            label_2
            velodyne
        train.txt
        val.txt
```
\* denotes generated folders

### Pre-trained ResNet-101
Download the pre-trained ResNet-101 model (faster_rcnn_resnet101_kitti) from the Tensorflow Object Detection API [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), and extract it in `data/pretrained` as `data/pretrained/faster_rcnn_resnet101_kitti_2018_01_28`

### Model Configuration
A sample configuration for training is in `src/monopsr/configs`. You can train using the example configs, or modify an existing configuration.

### Run Training
To start training, run the following:
```
python src/monopsr/experiments/run_training.py --config_path='src/monopsr/configs/monopsr_model_000.yaml' 
```

### Run Evaluation
To start evaluation, run the following:
```
python src/monopsr/experiments/run_evaluation.py --config_path='src/monopsr/configs/monopsr_model_000.yaml'
```
Note, we primarily use this script to determine metrics on the centroid and point cloud 
estimation. This is not used to obtain the validation results in the paper since it uses some 
ground truth boxes. To get the validation results in the paper we use `run_inference.py`.

### Run Inference
To start inference, run the following:
```
python src/monopsr/experiments/run_inference.py --config_path='src/monopsr/configs/monopsr_model_000
.yaml' --default_ckpt_num='100000' --data_split='val'
```
To calculate AP performance, follow the instructions in `scripts/offline_eval/kitti_native_eval`

## Contact
Please contact `kujason.ku@mail.utoronto.ca` or `alex.pon@mail.utoronto.ca` for any questions or issues.
