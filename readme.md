# CGC-Net: Cell Graph Convolutional Network for Grading of Colorectal Cancer Histology Images
by Yanning Zhou, Simon Graham, Navid Alemi Koohbanani, Muhammad Shaban, Pheng-Ann Heng and Nasir Rajpoot.

## Introduction
This repository is for our ICCVW2019 paper 'CGC-Net: Cell Graph Convolutional Network for Grading of Colorectal Cancer Histology Images'.

## Requirements
-   python 3.6.1
-   torch 1.1.0
-   torch-geometric 1.2.1
-   Other packages in requirements.txt

## Usage
prerequisite: your dataset images and cell/nuclei instance masks 
1. Clone the repository and set up the folders in the following structure:
```
 ├── code                   
 |   ├── CGC-Net
 |
 ├── data 
 |   ├── proto
 |        ├──mask (put the instance masks into this folder)    
 |             ├──"your-dataset"
 |                 ├──fold_1
 |                       ├──1_normal
 |                       ├──2_low_grade
 |                       ├──3_high_grade
 |                 ├──fold_2
 |                 ├──fold_3
 |
 |   ├── raw(put the images into this folder)	   
 |        ├──"your-dataset"
 |                 ├──fold_1
 |                       ├──1_normal
 |                       ├──2_low_grade
 |                       ├──3_high_grade
 |                 ├──fold_2
 |                 ├──fold_3
 ├── experiment	
 
 ```
2. Generate appearance features and distance tables.
 ```angular2html
cd CGC-Net/dataflow
python construct_feature_graph.py
```
You may need to change name and path in `class DataSetting` and `class GraphSetting(DataSetting)`.

3. To speed up training, we do sampling and fix the graph for each epoch before training start.
This step can be skipped when you want to do sampling inside dataloader by setting `dynamic_graph=True`.
```angular2html
cd CGC-Net/dataflow
python prepare_cv_dataset.py
```
You may need to change the sampling method, times and other parameters acccording to your need.

4. Train the model:
```angular2html
cd CGC-Net
sh parallel_train.sh
```

## Citation
If CGC-Net is useful for your research, please consider citing:
```angular2html
@inproceedings{zhou2019cgc,
  title={CGC-Net: Cell Graph Convolutional Network for Grading of Colorectal Cancer Histology Images},
  author={Zhou, Yanning and Graham, Simon and Koohbanani, Navid Alemi and Shaban, Muhammad and Heng, Pheng-Ann and Rajpoot, Nasir},
  booktitle={The IEEE International Conference on Computer Vision (ICCV) Workshops}, 
  year={2019}
}
```