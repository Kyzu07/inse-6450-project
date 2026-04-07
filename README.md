# INSE 6450: Human Motion Prediction using Masked Motion Completion

Implementation of **HumanMAC** (Chen et al., ICCV 2023) for the INSE 6450 course project at Concordia University.

**Student:** Shah Imran | **ID:** 40351654

---

## Overview

This project implements 3D human motion prediction by framing it as a **masked completion problem in DCT frequency space**. Given 25 observed frames (1s @ 50Hz), the model predicts 100 future frames (4s) using a diffusion-based transformer on the Human3.6M dataset.

---


## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/Kyzu07/inse-6450-project
cd inse-6450-project
pip install -r requirement.txt
```

**2. Download datasets**

- **Human3.6M**: Follow the instructions at [http://vision.imar.ro/human3.6m](http://vision.imar.ro/human3.6m). Place processed files in `data/`.
- **HumanEva-I**: Available at [http://humaneva.is.tue.mpg.de](http://humaneva.is.tue.mpg.de). Place in `data/`.

Pre-processed versions compatible with this repo can be obtained following the [DLow data preparation guide](https://github.com/Khrylx/DLow).

---

## Training

**Via notebook (recommended):**
```bash
jupyter notebook notebooks/training_humanmac.ipynb
```

---

## Results (91 epochs, RTX 4090)

| Metric | HumanMAC (paper) | HumanMAC (Our Run) |
|--------|------------------|--------------------|
| ADE ↓  | 0.369            | 0.4074                  |
| FDE ↓  | 0.480            | 0.5182                  |
| APD ↑  | 6.301            | 6.8759                  |

Training: **57.1s/epoch** · **28.4M parameters** · Best val loss: **0.03608** @ epoch 82

---

## Reference
```bibtex
@inproceedings{chen2023humanmac,
  title={HumanMAC: Masked Motion Completion for Human Motion Prediction},
  author={Chen, Ling-Hao and Zhang, Jiawei and Li, Yewen and Pang, Yiren and Xia, Xiaobo and Liu, Tongliang},
  booktitle={ICCV},
  year={2023}
}
```
