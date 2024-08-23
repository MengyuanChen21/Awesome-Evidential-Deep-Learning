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
| 9 | 2024 | WACV | DUC-VAR | Evidential uncertainty quantification: A variance-based perspective | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/WACV2024/papers/Duan_Evidential_Uncertainty_Quantification_A_Variance-Based_Perspective_WACV_2024_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/KerryDRX/EvidentialADA)](https://github.com/KerryDRX/EvidentialADA) |

## EDL Enhanced Machine Learning

### Weakly Supervised Learning
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 1 | 2021 | ICRA | FGR | Fgr: Frustum-aware geometric reasoning for weakly supervised 3d vehicle detection | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9561245) | [![GitHub stars](https://img.shields.io/github/stars/weiyithu/FGR)](https://github.com/weiyithu/FGR) 
| 2 | 2022 | ICPR | Map-gen | Map-gen: An automated 3d-box annotation flow with multimodal attention point generator | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9956415) | -
| 3 | 2022 | ECCV | DELU | Dual-evidential learning for weakly-supervised temporal action localization | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640190.pdf) | [![GitHub stars](https://img.shields.io/github/stars/MengyuanChen21/ECCV2022-DELU)](https://github.com/MengyuanChen21/ECCV2022-DELU) 
| 4 | 2022 | ECCV | OpenVAD | Towards open set video anomaly detection | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940387.pdf) | [![GitHub stars](https://img.shields.io/github/stars/YUZ128pitt/Towards-OpenVAD)](https://github.com/YUZ128pitt/Towards-OpenVAD) 
| 5 | 2023 | CVPR | CELL | Cascade Evidential Learning for Open-world Weakly-supervised Temporal Action Localization | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Cascade_Evidential_Learning_for_Open-World_Weakly-Supervised_Temporal_Action_Localization_CVPR_2023_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/MengyuanChen21/CVPR2023-OWTAL)](https://github.com/MengyuanChen21/CVPR2023-OWTAL) 
| 6 | 2023 | CVPR | CMPAE | Collecting cross-modal presence-absence evidence for weakly-supervised audio-visual event perception | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_Collecting_Cross-Modal_Presence-Absence_Evidence_for_Weakly-Supervised_Audio-Visual_Event_Perception_CVPR_2023_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/MengyuanChen21/CVPR2023-CMPAE)](https://github.com/MengyuanChen21/CVPR2023-CMPAE) 
| 7 | 2023 | TPAMI | VEL | Vectorized Evidential Learning for Weakly-supervised Temporal Action Localization | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10238828) | -
| 8 | 2024 | ICRA | MEDL-U | MEDL-U: Uncertainty-aware 3D Automatic Annotator based on Evidential Deep Learning | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2309.09599) | [![GitHub stars](https://img.shields.io/github/stars/paathelb/MEDL-U)](https://github.com/paathelb/MEDL-U) 
| 9 | 2024 | ICML | MIREL | Weakly-Supervised Residual Evidential Learning for Multi-Instance Uncertainty Estimation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://raw.githubusercontent.com/mlresearch/v235/main/assets/liu24ac/liu24ac.pdf) | [![GitHub stars](https://img.shields.io/github/stars/liupei101/MIREL)](https://github.com/liupei101/MIREL) 

### Transfer Learning
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 1 | 2021 | ICML | KD3A | KD3A: Unsupervised Multi-Source Decentralized Domain Adaptation via Knowledge Distillation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2011.09757) | [![GitHub stars](https://img.shields.io/github/stars/FengHZ/KD3A)](https://github.com/FengHZ/KD3A) 
| 2 | 2021 | CVPR | DECISION | Unsupervised multi-source domain adaptation without access to source data | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9577758) | [![GitHub stars](https://img.shields.io/github/stars/driptaRC/DECISION)](https://github.com/driptaRC/DECISION) 
| 3 | 2022 | ICLR | Sphereface2 | Sphereface2: Binary classification is all you need for deep face recognition | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2108.01513v1) | [![GitHub stars](https://img.shields.io/github/stars/ydwen/opensphere/tree/OpenSphere_v0)](https://github.com/ydwen/opensphere/tree/OpenSphere_v0) 
| 4 | 2022 | OpenReview | - | Modeling Unknown Semantic Labels as Uncertainty in the Prediction: Evidential Deep Learning for Class-Incremental Semantic Segmentation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openreview.net/pdf?id=-BBL3b4Tqfo) | -
| 5 | 2022 | arXiv | BEL | Bayesian evidential learning for few-shot classification | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2207.13137) | -
| 6 | 2022 | AAAI | TNT | Evidential neighborhood contrastive learning for universal domain adaptation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/download/20575/20334) | -
| 7 | 2023 | ICCV | CEDL | Continual Evidential Deep Learning for Out-of-Distribution Detection | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/Aguilar_Continual_Evidential_Deep_Learning_for_Out-of-Distribution_Detection_ICCVW_2023_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/Eaaguilart/cedl)](https://github.com/Eaaguilart/cedl) 
| 8 | 2024 | CVPR | FEAL | Think Twice Before Selection: Federated Evidential Active Learning for Medical Image Analysis with Domain Shifts | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Think_Twice_Before_Selection_Federated_Evidential_Active_Learning_for_Medical_CVPR_2024_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/JiayiChen815/FEAL)](https://github.com/JiayiChen815/FEAL) 
| 9 | 2024 | TPAMI | EAAF | Evidential Multi-Source-Free Unsupervised Domain Adaptation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10420513) | [![GitHub stars](https://img.shields.io/github/stars/SPIresearch/EAAF)](https://github.com/SPIresearch/EAAF) 
| 10 | 2024 | OpenReview | BKD | Bayesian Knowledge Distillation for Online Action Detection | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openreview.net/pdf?id=8iojQVLLWb) | -
| 11 | 2024 | arXiv | UGA | Uncertainty-Guided Alignment for Unsupervised Domain Adaptation in Regression | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2401.13721) | -
| 12 | 2024 | CVPR | MADA | Revisiting the Domain Shift and Sample Uncertainty in Multi-source Active Domain Transfer | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Revisiting_the_Domain_Shift_and_Sample_Uncertainty_in_Multi-source_Active_CVPR_2024_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/wannature/Detective-A-Dynamic-Integrated-Uncertainty-Valuation-Framework)](https://github.com/wannature/Detective-A-Dynamic-Integrated-Uncertainty-Valuation-Framework)  

### Active Learning
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 1 | 2022 | MICCAI | CSEAL | Consistency-based semi-supervised evidential active learning for diagnostic radiograph classification | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2209.01858) | -
| 2 | 2023 | ICLR | MEH + HUA | Active learning for object detection with evidential deep learning and hierarchical uncertainty aggregation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openreview.net/pdf?id=MnEjsw-vj-X) | [![GitHub stars](https://img.shields.io/github/stars/MoonLab-YH/AOD_MEH_HUA)](https://github.com/MoonLab-YH/AOD_MEH_HUA) 
| 3 | 2023 | ICLR | EDAL | Evidential uncertainty and diversity guided active learning for scene graph generation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openreview.net/pdf?id=xI1ZTtVOtlz) | -
| 4 | 2023 | NeurIPS | ADL | Multifaceted uncertainty estimation for label-efficient deep learning | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.neurips.cc/paper_files/paper/2020/file/c80d9ba4852b67046bee487bcd9802c0-Paper.pdf) | -
| 5 | 2023 | Deep Learning Applications | Deal | Deal: Deep evidential active learning for image classification | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2007.11344) | -
| 6 | 2024 | CVPR | --- | Evidential Active Recognition: Intelligent and Prudent Open-World Embodied Perception |[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_Evidential_Active_Recognition_Intelligent_and_Prudent_Open-World_Embodied_Perception_CVPR_2024_paper.pdf) | =


### Multi-View Classification
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 7 | 2022 | TPAMI | TMC/ETMC | Trusted multi-view classification with dynamic evidential fusion
| [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2204.11423) | [![GitHub stars](https://img.shields.io/github/stars/hanmenghan/TMC)](https://github.com/hanmenghan/TMC)
| 8 | 2023 | CVPR | UIMC | Exploring and exploiting uncertainty for incomplete multi-view classification
|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_Exploring_and_Exploiting_Uncertainty_for_Incomplete_Multi-View_Classification_CVPR_2023_paper.pdf) | -
| 9 | 2023 | ICLR | VS-FLEF | MULTI-VIEW DEEP EVIDENTIAL FUSION NEURAL NETWORK FOR ASSESSMENT OF SCREENING MAMMOGRAMS
| [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openreview.net/pdf?id=snjmwYRuqh) | [![GitHub stars](https://img.shields.io/github/stars/)](---)
| 10 | 2023 | MICCAI | --- | A reliable and interpretable framework of multi-view learning for liver fibrosis staging
| [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2306.12054) | -
| 11 | 2022 | IEEE Transactions on Industrial Informatics | EMDL | Uncertainty-aware multiview deep learning for internet of things applications
|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=9906001&ref=aHR0cHM6Ly9pZWVleHBsb3JlLmllZWUub3JnL3N0YW1wL3N0YW1wLmpzcD90cD0mYXJudW1iZXI9OTkwNjAwMQ==) | -
| 12 | 2023 | Remote Sensing | --- | Credible remote sensing scene classification using evidential fusion on aerial-ground dual-view images
|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-15-01546/article_deploy/remotesensing-15-01546-v3.pdf?version=1678846764) | -
| 13 | 2024 | AAAI | --- | Reliable conflictive multi-view learning
| [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/view/29546/30911) | [![GitHub stars](https://img.shields.io/github/stars/jiajunsi/RCML)(https://github.com/jiajunsi/RCML)

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

