# Awesome Evidential Deep Learning
A curated publication list on evidential deep learning.

This repository was built to facilitate navigating the mainstream on **evidential deep learning***.  

*Last updated: 2024/08*

##

## Table of Contents
- [Theoretical Explorations](#theoretical-explorations)
    - [Surveys](#surveys)
    - [Reformulating evidence collection process](#reformulating-evidence-collection-process)
    - [Improving uncertainty estimation via OOD samples](#improving-uncertainty-estimation-via-OOD-samples)
    - [Delving into different training strategies](#delving-into-different-training-strategies)
    - [Deep evidential regression](#deep-evidential-regression)
- [EDL Enhanced Machine Learning](#edl-enhanced-machine-learning)
    - [Weakly Supervised Learning](#open-set-recognition)
    - [Transfer Learning](#transfer-learning)
    - [Active Learning](#active-learning)
    - [Multi-View Classification](#multi-view-classification)
    - [Multi-label Learning](#multi-label-learning)
    - [Reinforcement Learning](#reinforcement)
    - [Graph Neural Network](#graph-neural-network)
- [EDL in Downstream Applications](#edl-in-downstream-applications)
    - [Computer Vision](#computer-vision)
    - [Natural Language Processing](#natural-language-processing)
    - [Cross-modal Learning](#cross-modal-learning)
    - [Automatic Driving](#automatic-driving)
    - [EDL in the Open-World](#edl-in-the-open-world)
    - [EDL for Science](#edl-for-science)
- [Feedback](#feedback)

##

## Theoretical Explorations
### Surveys
| ID | Year | Venue |  Title   |   PDF   |
|:--:|:----:|:-----:|:--------:|:-------:|
| 1 | 2022 | arXiv | A survey on uncertainty reasoning and quantification for decision making: Belief theory meets deep learning | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2206.05675) |
| 2 | 2023 | TMLR | Prior and posterior networks: A survey on evidential deep learning methods for uncertainty estimation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2110.03051) |
| 3 | 2024 | arXiv |  A comprehensive survey on evidential deep learning and its applications | |


### Reformulating evidence collection process
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 1 | 2017 | NeurIPS | EDL | Evidential deep learning to quantify classification uncertainty | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.neurips.cc/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/dougbrion/pytorch-classification-uncertainty)](https://github.com/dougbrion/pytorch-classification-uncertainty) 
| 2 | 2023 | ICML | RED | Learn to accumulate evidence from all training samples: Theory and practice | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.mlr.press/v202/pandey23a/pandey23a.pdf) | [![GitHub stars](https://img.shields.io/github/stars/pandeydeep9/EvidentialResearch2023)](https://github.com/pandeydeep9/EvidentialResearch2023) 
| 3 | 2023 | ICML | I-EDL | Uncertainty estimation by fisher information-based evidential deep learning | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.mlr.press/v202/deng23b/deng23b.pdf) | [![GitHub stars](https://img.shields.io/github/stars/danruod/IEDL)](https://github.com/danruod/IEDL) 
| 4 | 2023 | AAAI | - | Post-hoc uncertainty learning using a dirichlet meta-model | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/view/26167/25939) | -
| 5 | 2024 | ICLR | R-EDL | R-EDL: Relaxing Nonessential Settings of Evidential Deep Learning | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openreview.net/pdf?id=Si3YFA641c) | [![GitHub stars](https://img.shields.io/github/stars/MengyuanChen21/ICLR2024-REDL)](https://github.com/MengyuanChen21/ICLR2024-REDL) 
| 6 | 2024 | ICLR | Hyper EDL | Hyper Evidential Deep Learning to Quantify Composite Classification Uncertainty | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2404.10980) | [![GitHub stars](https://img.shields.io/github/stars/Hugo101/HyperEvidentialNN)](https://github.com/Hugo101/HyperEvidentialNN) 

### Improving uncertainty estimation via OOD samples
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 1 | 2019 | arXiv | - | Quantifying classification uncertainty using regularized evidential neural networks | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/1910.06864) | [Dropbox](https://www.dropbox.com/sh/uhonftulu9x2xa9/AABZxzeraWN8SYHh9N_10QoTa?dl=0) 
| 2 | 2020 | AAAI | GEN | Uncertainty-aware deep classifiers using generative models | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://aaai.org/ojs/index.php/AAAI/article/view/6015/5871) | [Ipynb](https://muratsensoy.github.io/gen.html) 
| 3 | 2021 | AAAI | WENN | Multidimensional uncertainty-aware evidential neural networks | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/download/16954/16761) | [![GitHub stars](https://img.shields.io/github/stars/snowood1/wenn)](https://github.com/snowood1/wenn) 
| 4 | 2023 | Sci Rep | m-EDL | Learning and predicting the unknown class using evidential deep learning | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://www.nature.com/articles/s41598-023-40649-w.pdf) | [![GitHub stars](https://img.shields.io/github/stars/naga0862/m-EDL)](https://github.com/naga0862/m-EDL) 
| 5 | 2023 | KDD | - | Knowledge from uncertainty in evidential deep learning | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2310.12663) | -

### Delving into different training strategies
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 1 | 2019 | AABI | BEDL | Bayesian evidential deep learning with PAC regularization | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/1906.00816) | -
| 2 | 2021 | WACV | Risk EDL | Misclassification risk and uncertainty quantification in deep classifiers | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](http://openaccess.thecvf.com/content/WACV2021/papers/Sensoy_Misclassification_Risk_and_Uncertainty_Quantification_in_Deep_Classifiers_WACV_2021_paper.pdf) | -
| 3 | 2022 | arXiv | TEDL | Tedl: A two-stage evidential deep learning method for classification uncertainty quantification | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2209.05522) | -
| 4 | 2022 | NeurIPS Workshop | Hybrid-EDL | Hybrid-edl: Improving evidential deep learning for uncertainty quantification on imbalanced data | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openreview.net/pdf?id=Nr1RDmAx-Qm) | [![GitHub stars](https://img.shields.io/github/stars/XTxiatong/Hybrid-EDL)](https://github.com/XTxiatong/Hybrid-EDL) 
| 5 | 2022 | CVPR | Units-ML | Multidimensional belief quantification for label-efficient meta-learning | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/CVPR2022/papers/Pandey_Multidimensional_Belief_Quantification_for_Label-Efficient_Meta-Learning_CVPR_2022_paper.pdf) | -
| 6 | 2022 | ICLR | ETP | Evidential Turing Processes | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2106.01216) | [![GitHub stars](https://img.shields.io/github/stars/ituvisionlab/EvidentialTuringProcess)](https://github.com/ituvisionlab/EvidentialTuringProcess) |






### Deep evidential regression
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 1 | 2020 | NeurIPS | DER | Deep evidential regression | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.neurips.cc/paper_files/paper/2020/file/aab085461de182608ee9f607f3f7d18f-Paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/aamini/evidential-deep-learning)](https://github.com/aamini/evidential-deep-learning) | 
| 2 | 2021 | arXiv | MDER | Multivariate deep evidential regression | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2104.06135) | -
| 3 | 2021 | NeurIPS | MoNIG | Trustworthy multimodal regression with mixture of normal-inverse gamma distributions | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.neurips.cc/paper/2021/file/371bce7dc83817b7893bcdeed13799b5-Paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/MaHuanAAA/MoNIG)](https://github.com/MaHuanAAA/MoNIG) |
| 4 | 2022 | AAAI | MT-ENet | Improving evidential deep learning via multi-task learning | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/download/20759/20518) | [![GitHub stars](https://img.shields.io/github/stars/deargen/MT-ENet)](https://github.com/deargen/MT-ENet) |
| 5 | 2023 | AAAI | - | The unreasonable effectiveness of deep evidential regression | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/download/26096/25868) | [![GitHub stars](https://img.shields.io/github/stars/pasteurlabs/unreasonable_effective_der)](https://github.com/pasteurlabs/unreasonable_effective_der) |
| 6 | 2023 | AAAI | ECNP | Evidential conditional neural processes | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/view/26125/25897) | -
| 7 | 2024 | AAAI | - | The evidence contraction issue in deep evidential regression: Discussion and solution | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/view/30172/32080) | [![GitHub stars](https://img.shields.io/github/stars/yuelfei/evi_con)](https://github.com/yuelfei/evi_con) |
| 8 | 2024 | AAAI | UR-ERN | Uncertainty regularized evidential regression | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/download/29583/30981) | -
|  | 2024 | WACV | DUC-VAR | Evidential uncertainty quantification: A variance-based perspective | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/WACV2024/papers/Duan_Evidential_Uncertainty_Quantification_A_Variance-Based_Perspective_WACV_2024_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/KerryDRX/EvidentialADA)](https://github.com/KerryDRX/EvidentialADA) |

## EDL Enhanced Machine Learning

### Weakly Supervised Learning
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|

### Transfer Learning
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|

### Active Learning
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|

### Multi-View Classification
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|

### Multi-label Learning
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|

### Reinfocement Learning
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|

### Graph Neural Network
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 1 | 2020 | NeurIPS | GKDE | Uncertainty aware semi-supervised learning on graph data | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.neurips.cc/paper/2020/file/968c9b4f09cbb7d7925f38aea3484111-Paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/zxj32/uncertainty-GNN)](https://github.com/zxj32/uncertainty-GNN) |

## EDL in Downstream Applications

### Computer Vision
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|

### Natural Language Processing
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|

### Cross-modal Learning
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|

### Automatic Driving
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|

### EDL in the Open-World
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|

### EDL for Science
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|

## Feedback

If you have any suggestions or find missing papers, please feel free to contact me.

- [e-mail](mailto:chenmengyuan2021@ia.ac.cn)
- [pull request](https://github.com/MengyuanChen21/Awesome-Evidential-Deep-Learning/pulls)

