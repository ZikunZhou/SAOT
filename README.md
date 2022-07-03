## SAOT

The official implementation of the paper [**Saliency-Associated Object Tracking**](https://arxiv.org/abs/2108.03637) (accepted by ICCV2021).

![SAOT_Framework](SAOT_framework.jpg)

### Updates

Support PyTorch1.7 and CUDA11.0.

Support DDP.

Update the training settings for better performance. 

Training our model with the new settings using a single GPU, the performance on LaSOT reaches 0.631 (AUC).

### Performance reported in the paper
| Dataset | LaSOT (AUC)| GOT-10K (AO)| OTB2015 (AUC)| VOT2018 (EAO) |
|---|---|---|---|---|
|**SAOT**|**0.616**|**0.640**|**0.714**|**0.501**|
|Ocean|0.560|0.611|0.684|0.489|
|DiMP|0.568|0.611|0.686|0.440|

### Real-time Speed
SAOT runs at about **29 FPS** on an RTX 2080 GPU with PyTorch 1.1 and CUDA 10.

### Model Zoo & Raw Results
The pre-trained model and raw results are provided in [model zoo](https://drive.google.com/drive/folders/1T5F4JsZ-P-vzzUr5KXJxw853LTUO_lmb?usp=sharing) and [raw results](https://drive.google.com/drive/folders/1_x6mlr0rVbF4sUuasCgIxYINvOqqHCJq?usp=sharing), respectively. 


### Installation

#### Clone the GIT repository.  
```bash
git clone https://github.com/ZikunZhou/SAOT.git
```
   
#### Clone the submodules.  
In the repository directory, run the commands:  
```bash
git submodule update --init  
```
#### Install dependencies
Please note that using a new version pytorch may cause unexpected issues.
Following the [detailed installation instructions](INSTALL.md) to install the dependencies.


### Testing
Download the pre-trained networks.
Activate the conda environment and run the script pytracking/test_saot_fs.py.  
```bash
conda activate SAOT
cd pytracking
python test_saot_fs.py saot saot_otb --dataset OTB2015
```

### Training
Download the training datasets.
Activate the conda environment and run the script train.py.
```bash
conda activate SAOT
python train.py --train_module dimp --train_name saot --mode multiple --nproc_per_node 1
```

### Citation
Please cite the following publication, if you find the code helpful in your research.
```
@InProceedings{Zhou_2021_ICCV,
    author    = {Zhou, Zikun and Pei, Wenjie and Li, Xin and Wang, Hongpeng and Zheng, Feng and He, Zhenyu},
    title     = {Saliency-Associated Object Tracking},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {9866-9875}
}
```

### Acknowledgments
Thanks for the [PyTracking](https://github.com/visionml/pytracking) and [Pysot](https://github.com/STVIR/pysot.git) libraries, which helps us to quickly implement our ideas. 

Thanks for Kaige Mao for the helps about the updates.

### Contact
Please feel free to contact me (Email: zikunzhou@163.com).

