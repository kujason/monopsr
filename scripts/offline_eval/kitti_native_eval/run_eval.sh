#!/bin/bash

set -e

#echo $1  # eval_script_dir
#echo $2  # checkpoint_name
#echo $3  # score_threshold
#echo $4  # global step
#echo $5  # prediction_dir
#echo $6  # results dir

eval_script_dir=$1
checkpoint_name=$2
score_threshold=$3
global_step=$4
prediction_dir=$5
results_dir=$6

cd ${eval_script_dir}
#echo "${global_step}" | tee -a ${results_dir}/${checkpoint_name}_results_${score_threshold}.txt
./evaluate_object_3d_offline ~/Kitti/object/training/label_2/ ${prediction_dir} | tee -a ${results_dir}/${checkpoint_name}_results_${score_threshold}.txt
