# Installation

This document contains detailed instructions for installing the necessary dependencies for PyTracking. The instrustions have been tested on an Ubuntu 18.04 system with an RTX2080 GPU.

### Requirements  
* Conda installation with Python 3.7. If not already installed, install from https://www.anaconda.com/distribution/.
* CUDA 10.0, pytoch 1.1.0, Nvidia GPU. We found that this code is not compatible with later versions of pytorch. We will fix it as soon as possible.

## Step-by-step instructions  
#### Create and activate a conda environment
```bash
conda create --name SAOT python=3.7
conda activate SAOT
```

#### Install PyTorch  
Install PyTorch1.1 with cuda10.
(PyTorch1.7 with cuda11.0 is supported after updating)

```bash
conda install pytorch=1.1 torchvision cudatoolkit=10.0 -c pytorch
```

**Note:**  
- For more details about PyTorch installation, see https://pytorch.org/get-started/previous-versions/.  

#### Install matplotlib, pandas, tqdm, opencv, scikit-image, visdom, tikzplotlib, gdown, and tensorboad  
```bash
conda install matplotlib pandas tqdm
pip install opencv-python visdom tb-nightly scikit-image tikzplotlib gdown
```


#### Install the coco toolkit  
If you want to use COCO dataset for training, install the coco python toolkit. You additionally need to install cython to compile the coco toolkit.
```bash
conda install cython
pip install pycocotools
```


#### Install ninja-build for Precise ROI pooling  
To compile the Precise ROI pooling module (https://github.com/vacancy/PreciseRoIPooling), you may additionally have to install ninja-build.
```bash
sudo apt-get install ninja-build
```
In case of issues, we refer to https://github.com/vacancy/PreciseRoIPooling.  


#### Install jpeg4py  
In order to use [jpeg4py](https://github.com/ajkxyz/jpeg4py) for loading the images instead of OpenCV's imread(), install jpeg4py in the following way,  
```bash
sudo apt-get install libturbojpeg
pip install jpeg4py 
```

**Note:** The first step (```sudo apt-get install libturbojpeg```) can be optionally ignored, in which case OpenCV's imread() will be used to read the images. However the second step is a must.  

In case of issues, we refer to https://github.com/ajkxyz/jpeg4py.  


#### Setup the environment  
Create the default environment setting files. 
```bash
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

You can modify these files to set the paths to datasets, results paths etc.  
