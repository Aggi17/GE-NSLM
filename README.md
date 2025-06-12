# Unveiling Implicit Deceptive Patterns in Fake News: A Neuro-Symbolic Approach with LVLMs

This is the official implementation of the paper "Unveiling Implicit Deceptive Patterns in Fake News: A Neuro-Symbolic Approach with LVLMs".

## Overview

This repository contains the code for a neuro-symbolic approach to fake news detection using Large Vision-Language Models (LVLMs). The approach combines neural networks with symbolic reasoning to identify implicit deceptive patterns in fake news content.

## Requirements
 Details see requirements.txt

- Python 3.7+
- PyTorch 1.13.0+
- Transformers 4.20.1+
- TensorFlow 2.11.0+
- CUDA 11.7+ (for GPU acceleration)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd GE-NSLM24
```

2. Install dependencies:
```bash
cd src/check_client
pip install -r requirements.txt
```

## Data

- You can refer to the papers released the Weibo and Fakeddit dataset for offical data.
- Here we provide Pre-processed 5-fold text data of Fakeddit-sub in the `data/fesub folder.

Place the dataset in the following structure:
```
data/
├── fakeddit/
│   ├── train_img/
│   ├── val_img/
│   ├── test_img/
│   └── [other data files]
```

## Usage

### Training

1. Set the project home directory:
```bash
export PJ_HOME=/path/to/your/project
```

2. Make the training script executable:
```bash
chmod +x train_roberta.sh
```

3. Run training:
```bash
sh train_roberta.sh
```

### Training Parameters

The training script accepts the following parameters:
- `lambda`: Logic lambda parameter (default: 0.5)
- `prior`: Prior type - uniform/nli/random (default: random)
- `mask`: Mask rate (default: 0.0)

Example:
```bash
sh train_roberta.sh 0.5 random 0.0
```



## Citation

If you use this code, please cite our paper
''Unveiling Implicit Deceptive Patterns in Fake News: A Neuro-Symbolic Approach with LVLMs''


## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contact

For questions and issues, please contact: [aggi19177@gmail.com] 




* Notes: 
    - You can find the output data in the `out` folder specified in the config file.
    - Since the probability of the four deceptive patterns appearing at the same time in actual situations is very small, in the experiment we set the y corresponding to the situation where all three deceptive modes exist to 2 (that is, a category that does not exist in the dataset), so the decoder finally becomes a three-category prediction
  
##Acknowledgment

Our implementation is mainly based on follows. Thanks for their authors. 
https://github.com/jiangjiechen/LOREN