# FUNIT_tensorflow
Tensorflow Implementation of FUNIT: Few-Shot Unsupervised Image-to-Image Translation

<p align="center"> <img src="./imgs/animal.gif" width="92%"> </p>

--------------------------------------------
[Paper](https://arxiv.org/abs/1905.01723) | [Pytorch Code](https://github.com/NVlabs/FUNIT)

## Dependencies

* [Python 3.6](https://www.python.org/download/releases/3.6/)
* [Tensorflow 1.13](https://github.com/tensorflow/tensorflow)
* [numpy](http://www.numpy.org/)

## Usage

- Clone this repo:
  ```bash
  git clone https://github.com/zhangqianhui/GazeCorrection.git
  ```
- Download the NewGaze dataset

  Download the tar of NewGaze dataset from [Google Driver Linking](https://drive.google.com/open?id=1lYzpKdShN68RJGxRF1JgXnW-ved0F-mJ).
  
  ```bash
  cd your_path
  unzip NewGazeData.tar
  ```

- Pretraining Model

  We have provided the self-guided pretraining model in directory: ./sg_pre_model_g

- Train this model using the your parameter

  (1)Please edit the config.py file to select the proper hyper-parameters.
  
  (2)Change the "base_path" to "your_path" of NewGaze dataset.
  
  Then
  
  ```bash
  python main.py 
  ```

## Our results

- Results on Flowers dataset 

 ![](img/exp2.jpg)
 
- Results on Animals dataset

 ![](img/exp1.jpg)

# Reference code

- [Gazecorrection](https://github.com/zhangqianhui/GazeCorrection)

- [FUNIT pytorch](https://github.com/NVlabs/FUNIT)

- [AttGAN_tensorflow](https://github.com/LynnHo/AttGAN-Tensorflow)


