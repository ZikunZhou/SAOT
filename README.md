# SAOT
The official implementation of the paper [**Saliency-Associated Object Tracking**](https://arxiv.org/abs/2108.03637) (ICCV2021).

![SAOT_Framework](SAOT_framework.jpg)
## Highlights

### Strong Performance
| Dataset | LaSOT (AUC)| GOT-10K (AO)| OTB2015 (AUC)| VOT2018 (EAO) |
|---|---|---|---|---|
|**SAOT**|**0.616**|**0.640**|**0.714**|**0.501**|
|Ocean|0.560|0.611|0.684|0.489|
|DiMP|0.568|0.611|0.686|0.440|

### Real-time Speed
SAOT runs at about **29 FPS** on an RTX 2080 GPU.

## [Model Zoo](MODEL_ZOO.md)
The tracker models trained using PyTracking, along with their results on standard tracking 
benchmarks are provided in the [model zoo](MODEL_ZOO.md). 


## Installation

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
Following this instructions [detailed installation instructions](INSTALL.md) to install the dependencies.


## Testing
Activate the conda environment and run the script pytracking/run_webcam.py to run ATOM using the webcam input.  
```bash
conda activate pytracking
cd pytracking
python run_webcam.py dimp dimp50    
```

## Training


## Acknowledgments
Thanks for the [PyTracking](https://github.com/visionml/pytracking) and [Pysot](https://github.com/STVIR/pysot.git) libraris, which helps us to quickly implement our ideas.
