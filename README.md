# Kaggle-Sartorius-Cell-Instance-Segmentation---Multi-model-ensemble-

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
    using utils/COCO_dataset_generator.ipynb (run the notebook as instructed in it).
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
