# SoundNet_Pytorch
converting the pretrained tensorflow SoundNet model to pytorch

![from soundnet](https://camo.githubusercontent.com/0b88af5c13ba987a17dcf90cd58816cf8ef04554/687474703a2f2f70726f6a656374732e637361696c2e6d69742e6564752f736f756e646e65742f736f756e646e65742e6a7067)

# Introduction
The code is for converting the pretrained [tensorflow soundnet model](https://github.com/eborboihuc/SoundNet-tensorflow) to pytorch model. So no training code for SoundNet model. The pretrained pytorch soundnet model can be found [here](https://drive.google.com/file/d/1-PhHutIYV9Oi2DhDZL2h1Myu84oGLI81/view?usp=sharing).

# Prerequisites
1. tensorflow (cpu or gpu)
2. python 3.6 with numpy
3. pytorch 0.4+
4. weight file: google drive: https://drive.google.com/drive/folders/1zjNiuLgZ1cjCzF80P4mlYe4KSGGOFlta?usp=sharing; 百度网盘：链接：https://pan.baidu.com/s/1v_K2pJvo0KE38EZ__WZJWg 提取码：iz4h 


# How to use
1. prepare the code
```
git clone https://github.com/smallflyingpig/SoundNet_Pytorch.git
cd SoundNet_Pytorch
```
2. prepare the tensorflow soundnet model parameters. Download from [sound8.npy](https://drive.google.com/uc?export=download&id=0B9wE6h4m--wjR015M1RLZW45OEU), which is provided by [eborboihuc](https://github.com/eborboihuc/SoundNet-tensorflow), and save in the current folder.
3. install the prerequisites
4. run
```
python tf2pytorch.py --tf_param_path ./sound8.npy --pytorch_param_path ./sound8.pth
```
5. test the result

download input demo data from [demo.py](https://drive.google.com/uc?export=download&id=0B9wE6h4m--wjcEtqQ3VIM1pvZ3c) and save to the current folder. We calculate the average feature errors at each convolution block (total 7 conv blocks) and the predictions for object/scene classification (2 layers), and output 9 error totally.
```
python check_layer.py --tf_param_path ./sound8.npy --pytorch_param_path ./sound8.pth --input_demo_data ./demo.npy
```
The expected output:
```
layer error:
[-1.3113022e-06, 0.0, 0.0, 0.0, 1.4901161e-08, 0.0, -6.9849193e-10, 4.7683716e-07, 7.1525574e-07]
```
This indicates the success of our model conversion.

6. extract features
after the pytorch model is got(save as ./sound8.pth), run the following command to extract features:
```
python example.py
```
# Acknowledgments
Code for soundnet tensorflow model is ported from [soundnet_tensorflow](https://github.com/eborboihuc/SoundNet-tensorflow). Thanks for his works!

# FAQs
Feel free to mail me(jiguo.li@vipl.ict.ac.cn or jgli@pku.edu.cn) if you have any questions about this project.
# reference
1. Yusuf Aytar, Carl Vondrick, and Antonio Torralba. "Soundnet: Learning sound representations from unlabeled video." Advances in Neural Information Processing Systems. 2016.

