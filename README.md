# Kaggle-Sartorius-Cell-Instance-Segmentation---Multi-model-ensemble-

https://www.kaggle.com/c/sartorius-cell-instance-segmentation


## Data Preparation
1) Download the competition dataset from: https://www.kaggle.com/c/sartorius-cell-instance-segmentation/data
2) Download LIVECell dataset from https://github.com/sartorius-research/LIVECell 
3) Unzip the files as follows:
```
├─ data
├───── LIVECell_dataset_2021
│      ├── images
│      ├── ann_coco_livecell_train.json
│      ├── ann_coco_livecell_val.json
│      └── ann_coco_livecell_test.json
├───── train
├───── train_semi_supervised
├───── train.csv
└───── test
```
4) Crate annotation files for: a) train folder  b) train_semi_supervised folder,
    using /utils/COCO_dataset_generator.ipynb (run the notebook as instructed in it).
5) The final data folder arangement as follows:
```
├─ data
├───── LIVECell_dataset_2021
│      ├── images
│      ├── ann_coco_livecell_train.json
│      ├── ann_coco_livecell_val.json
│      └── ann_coco_livecell_test.json
├───── train
├───── train_semi_supervised
├───── train.csv
├───── test
├───── ann_coco_sartorius_train_95_5
├───── ann_coco_sartorius_val_95_5
└───── ann_coco_semi
```
6) Download or clone this github folder, and place the data folder in it.

## Notes (before training): 
1) Our soloution use 4 detection models, and 1 segmentation model to train seperately in different jupyter notebooks.
2) To modify configurations/parameters, open and modify the config file coressponding to each model's folder according to https://mmdetection.readthedocs.io.
         (For example: /models/det/det1_cascade_rcnn_resnext/configs/config_det1_cascade_rcnn_resnext.py).
3) Must modify in each config file the next parmaeters (lines 1-11): main_dir, exp_name, wnb_username, wnb_project_name, livecell_or_sartorius (as instructed in it).
4) As fully explaind in our report, we first pretrained each model with Livecell dataset, to do so, choose in each config file if to train on Livecell dataset or competition dataset.

## Training
```
1) Open notebook:   /training_scripts/det/det1_cascade_rcnn_resnext.ipynb            (and run the notebook as instructed in it).
2) Open notebook:   /training_scripts/det/det2_cascade_rcnn_resnest.ipynb            (and run the notebook as instructed in it).
3) Open notebook:   /training_scripts/det/det3_faster_rcnn_swin.ipynb                (and run the notebook as instructed in it).
4) Open notebook:   /training_scripts/det/det4_softteacher_faster_rcnn_resnext.ipynb (and run the notebook as instructed in it).
5) Open notebook:   /training_scripts/seg/seg_upernet_swin.ipynb                     (and run the notebook as instructed in it).
```
## Inference/Test
```
Open: /inference_script/inference_det_and_seg.ipynb                    (and run the notebook as instructed in it).
```

The full test folder and its annotations is not available, so to test your results you must late submit it in Kaggle's competition page.

