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
 |   ├── raw(put the images into this folder))	   
 |        ├──"your-dataset"
 |                 ├──fold_1
 |                       ├──1_normal
 |                       ├──2_low_grade
 |                       ├──3_high_grade
 |                 ├──fold_2
 |                 ├──fold_3
 ├── experiment	
 
 