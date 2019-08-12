# kitti_native_eval

`evaluate_object_3d_offline.cpp`evaluates your KITTI detection locally on your own computer using your validation data selected from KITTI training dataset, with the following metrics:

- Average Precision In 2D Image Frame (AP)
- oriented overlap on image (AOS)
- Average Precision In BEV (AP)
- Average Precision In 3D (AP)

You can follow these instrutions to run this evaluation separately.

Install dependencies and build:
```
sudo apt-get install gnuplot gnuplot5
cd /kitti_native_eval
make
make eval_low_iou
```

Evaluation with 0.7, 0.5, 0.5 IoU for cars, pedestrians, and cyclists.
```bash
./evaluate_object_3d_offline groundtruth_dir result_dir
```

Evaluation with 0.5, 0.25, 0.25 IoU for cars, pedestrians, and cyclists.
```bash
./evaluate_object_3d_offline_low groundtruth_dir result_dir
```

- Detections should be in a `data` folder
- `groundtruth_dir` should be ~/Kitti/object/training/label_2

---

Note that you don't have to detect over all KITTI training data. The evaluator only evaluates samples whose result files exist.

- Results will appear per class in terminal for easy, medium and hard difficulties.
- Precision-Recall Curves will be generated and saved to a 'plot' dir.
