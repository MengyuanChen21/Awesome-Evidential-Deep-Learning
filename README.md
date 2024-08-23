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
| 3 | 2022 | OpenReview | - | Modeling Unknown Semantic Labels as Uncertainty in the Prediction: Evidential Deep Learning for Class-Incremental Semantic Segmentation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openreview.net/pdf?id=-BBL3b4Tqfo) | -
| 4 | 2022 | arXiv | BEL | Bayesian evidential learning for few-shot classification | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2207.13137) | -
| 5 | 2022 | AAAI | TNT | Evidential neighborhood contrastive learning for universal domain adaptation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/download/20575/20334) | -
| 6 | 2023 | ICCV | CEDL | Continual Evidential Deep Learning for Out-of-Distribution Detection | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/ICCV2023W/VCL/papers/Aguilar_Continual_Evidential_Deep_Learning_for_Out-of-Distribution_Detection_ICCVW_2023_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/Eaaguilart/cedl)](https://github.com/Eaaguilart/cedl) 
| 7 | 2024 | CVPR | FEAL | Think Twice Before Selection: Federated Evidential Active Learning for Medical Image Analysis with Domain Shifts | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Think_Twice_Before_Selection_Federated_Evidential_Active_Learning_for_Medical_CVPR_2024_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/JiayiChen815/FEAL)](https://github.com/JiayiChen815/FEAL) 
| 8 | 2024 | TPAMI | EAAF | Evidential Multi-Source-Free Unsupervised Domain Adaptation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10420513) | [![GitHub stars](https://img.shields.io/github/stars/SPIresearch/EAAF)](https://github.com/SPIresearch/EAAF) 
| 9 | 2024 | OpenReview | BKD | Bayesian Knowledge Distillation for Online Action Detection | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openreview.net/pdf?id=8iojQVLLWb) | -
| 10 | 2024 | arXiv | UGA | Uncertainty-Guided Alignment for Unsupervised Domain Adaptation in Regression | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2401.13721) | -
| 11 | 2024 | CVPR | MADA | Revisiting the Domain Shift and Sample Uncertainty in Multi-source Active Domain Transfer | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Revisiting_the_Domain_Shift_and_Sample_Uncertainty_in_Multi-source_Active_CVPR_2024_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/wannature/Detective-A-Dynamic-Integrated-Uncertainty-Valuation-Framework)](https://github.com/wannature/Detective-A-Dynamic-Integrated-Uncertainty-Valuation-Framework)  

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
| 1 | 2022 | TPAMI | TMC/ETMC | Trusted multi-view classification with dynamic evidential fusion | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2204.11423) | [![GitHub stars](https://img.shields.io/github/stars/hanmenghan/TMC)](https://github.com/hanmenghan/TMC)
| 2 | 2022 | IEEE Transactions on Industrial Informatics | EMDL | Uncertainty-aware multiview deep learning for internet of things applications |[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=9906001&ref=aHR0cHM6Ly9pZWVleHBsb3JlLmllZWUub3JnL3N0YW1wL3N0YW1wLmpzcD90cD0mYXJudW1iZXI9OTkwNjAwMQ==) | -
| 3 | 2023 | CVPR | UIMC | Exploring and exploiting uncertainty for incomplete multi-view classification |[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_Exploring_and_Exploiting_Uncertainty_for_Incomplete_Multi-View_Classification_CVPR_2023_paper.pdf) | -
| 4 | 2023 | ICLR | VS-FLEF | MULTI-VIEW DEEP EVIDENTIAL FUSION NEURAL NETWORK FOR ASSESSMENT OF SCREENING MAMMOGRAMS | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openreview.net/pdf?id=snjmwYRuqh) | -
| 5 | 2023 | MICCAI | --- | A reliable and interpretable framework of multi-view learning for liver fibrosis staging | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2306.12054) | -
| 6 | 2023 | Remote Sensing | --- | Credible remote sensing scene classification using evidential fusion on aerial-ground dual-view images |[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-15-01546/article_deploy/remotesensing-15-01546-v3.pdf?version=1678846764) | -
| 7 | 2024 | AAAI | --- | Reliable conflictive multi-view learning | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/view/29546/30911) | [![GitHub stars](https://img.shields.io/github/stars/jiajunsi/RCML)](https://github.com/jiajunsi/RCML)

### Multi-label Learning
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 1 | 2022 | ICASSP | PENet | Seed: Sound event early detection via evidential uncertainty |[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2202.02441) | -
| 2 | 2023 | CVPR | MULE | Open set action recognition via multi-label evidential learning |[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Open_Set_Action_Recognition_via_Multi-Label_Evidential_Learning_CVPR_2023_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/charliezhaoyinpeng/mule)](https://github.com/charliezhaoyinpeng/mule)
| 3 | 2023 | Artificial Intelligence | --- | DEED: DEep Evidential Doctor |[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://pdf.sciencedirectassets.com/271585/1-s2.0-S0004370223X00101/1-s2.0-S0004370223001650/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQCF8aXdRv35Ib765K8FTIiKkd21ALSY6XF1Y2X%2BOKpzqQIgPZOh9m7jJP98S%2FOjBkEYOPQnYnCiSulXJD4k6tVYcWIqvAUIov%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDCcEKgqBCA0s4Rye9iqQBfynHSWBtf2AFGq3CgqxEN%2F1sRWjrrw3WTYkNauYNWPkLm22Pt6reID%2BAHhVyNbNUBQfOvAai26iJ4SVcopW6vD7VDI23Cr82zu2YSTmg2piUo9Waktg9ufIZrmynmluHaHyte7VZut8JgSgpZKss%2BiWU8gZYLIifQHt1zA4apeuO7wvSDiFWy%2F6zFwQTN%2Fj53fjpC3e%2F0SWIaIBup2O8wu6hEEr3V7TFq93MavgcN5sNYZCUxYKpSvVMmWQcz6puhRL50s4r48%2Fw1gw%2FtMDPWmz8NZohRYuP%2BxCFnok0Q3zyM7yk2Kl%2FWYpQTlWho4nnqVcdGZviIgb87novspmgVLLJ6Y3yg%2B5RsO06QJL7OPz4alaVkXZGfCZeBFYJBymw00fFEFWvo%2FoltnNQLZGRyIbyC5PrZ0N71FG%2Bt9bFdO8EttHEseCpF2sn0wx014qek5DkgwCo9UkMoJf7k16iCN2ZgWpQG74%2FQIiVzUxRad8lY0OI95rqiQghT9%2FvYWBXqgSCHcIZtXK%2BRIgN28p%2BgZwRUEnR6kIw0mUon%2B%2B%2Fi37Mz4FAoq5g5jA1K1Ni44KmGxMueAqORdpbJqzF3mh8r%2F7fdggIyubU1SouAwRbFyl4Ipr0YRbQvFLpTsy%2BA114eProwTIzBheHrcwSisFU0x%2Bn3zSXq17%2FxMZLxYOERmd8UTtKakSo0ZvX%2BliE%2FRjrx7jk845dgamrobeWpr%2BYAjV68k99%2FKK8brThhan896EeYqXT596ljfvjne0ouGE6oeGVJWdmr%2F3TnBFdUb70%2B0%2F99Gtx2BMuq1zxW%2F3p3LwtaOMDgR9vqXroDsXCY8CiZBro1M84BanoA2oBqYtlM2N7vk22rXiRCp%2BaSiO64HGMLylobYGOrEBBI4PHtHAF%2FgIj0btooUSpWS9bX9UBoyWmKnv5w3rjMClettOQ1B9eMVjpbyG7lSq7KeQE7bekDCStP%2BTVN%2BNYFw5kcr1XbCMtknR39lFXkqIu5gqnEclIyg0QpwnMiB4j0Xn%2F%2FkkrFYXZRJDjFWes7Nk0fiJ82jPN1N0HkWdXL5t8MR3CxCZ%2BdkdLSfUpX2GWuBlp0GW%2FF9O%2BsJmZn9c6op5QuYuqGcd4T2Uic%2BicmXh&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240823T092147Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY4R4CQGM7%2F20240823%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=225277b5b055f9b92bb6d0473f33b65df512dd7038f25cf34899784f22383693&hash=4f178aa2fbd7896efc2d0ab863bf2fc51f6db62766a04d27a30f68106fdfff63&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0004370223001650&tid=spdf-9def5ee8-4fc0-4147-b509-43612b55c8cf&sid=6d6e6ef18ca979463228cac4f1e18cc3ecb4gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=030b5b065504505e06&rr=8b7a08d0e8a60f00&cc=cn) | [![GitHub stars](https://img.shields.io/github/stars/aaq109/DEED)](https://github.com/aaq109/DEED)
| 4 | 2023 | ICASSP | TMSDC | Towards trustworthy multi-label sewer defect classification via evidential deep learning |[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2210.13782) | -
| 5 | 2023 | ICASSP | MTENN | Multi-Label Temporal Evidential Neural Networks for Early Event Detection |[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=10096305&ref=aHR0cHM6Ly9pZWVleHBsb3JlLmllZWUub3JnL3N0YW1wL3N0YW1wLmpzcD90cD0mYXJudW1iZXI9MTAwOTYzMDU=) | -



### Reinfocement Learning
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 1 | 2023 | LCML | DTS-ERA | Deep temporal sets with evidential reinforced attentions for unique behavioral pattern discovery |[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.mlr.press/v202/wang23ab/wang23ab.pdf) | [![GitHub stars](https://img.shields.io/github/stars/wdr123/DTS_ERA)](https://github.com/wdr123/DTS_ERA)
| 2 | 2023 | NeurIPS | FGRM | Uncertainty estimation for safety-critical scene segmentation via fine-grained reward maximization |[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.neurips.cc/paper_files/paper/2023/file/71ec377d5df1fc61ee7770857820519b-Paper-Conference.pdf) | [![GitHub stars](https://img.shields.io/github/stars/med-air/FGRM)](https://github.com/med-air/FGRM)
### Graph Neural Network
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 1 | 2020 | NeurIPS | GKDE | Uncertainty aware semi-supervised learning on graph data | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.neurips.cc/paper/2020/file/968c9b4f09cbb7d7925f38aea3484111-Paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/zxj32/uncertainty-GNN)](https://github.com/zxj32/uncertainty-GNN) |

## EDL in Downstream Applications

### Computer Vision
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 1 | 2019 | MICCAI  | - | Quantifying and leveraging classification uncertainty for chest radiograph assessment | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://link.springer.com/content/pdf/10.1007/978-3-030-32226-7.pdf) | -
| 2 | 2021 | ICCV | DEAR | Evidential Deep Learning for Open Set Action Recognition | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/ICCV2021/papers/Bao_Evidential_Deep_Learning_for_Open_Set_Action_Recognition_ICCV_2021_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/Cogito2012/DEAR)](https://github.com/Cogito2012/DEAR) 
| 3 | 2022 | ACM MM | - | Uncertainty-aware 3d human pose estimation from monocular video | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://dl.acm.org/doi/pdf/10.1145/3503161.3547773) | -
| 4 | 2022 | Cognitive Computation | EviDCNN-3WC | Three-way image classification with evidential deep convolutional neural networks | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://link.springer.com/content/pdf/10.1007/s12559-021-09869-y.pdf) | -
| 5 | 2022 | CVPR | OpenTAL | Opental: Towards open set temporal action localization | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/CVPR2022/papers/Bao_OpenTAL_Towards_Open_Set_Temporal_Action_Localization_CVPR_2022_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/Cogito2012/OpenTAL)](https://github.com/Cogito2012/OpenTAL) 
| 6 | 2022 | ACM MM | - | Evidential Reasoning for Video Anomaly Detection | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://dl.acm.org/doi/pdf/10.1145/3503161.3548091) | -
| 7 | 2022 | PR | - | Uncertainty estimation for stereo matching based on evidential deep learning | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://www.sciencedirect.com/science/article/pii/S0031320321006749/pdfft?md5=596cd4f3382b1fd551b829296e3678de&pid=1-s2.0-S0031320321006749-main.pdf) | -
| 8 | 2022 | MICCAI | TBraTS | Tbrats: Trusted brain tumor segmentation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://link.springer.com/content/pdf/10.1007/978-3-031-16452-1.pdf) | -
| 9 | 2023 | IEEE RAL | EvPSNet | Uncertainty-aware panoptic segmentation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10068764) | -
| 10 | 2023 | ACM MM | FOOD | HSIC-based Moving Weight Averaging for Few-Shot Open-Set Object Detection | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://dl.acm.org/doi/pdf/10.1145/3581783.3611850) | [![GitHub stars](https://img.shields.io/github/stars/binyisu/food)](https://github.com/binyisu/food) 
| 11 | 2023 | MICCAI | UML | Uncertainty-informed mutual learning for joint medical image classification and segmentation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://link.springer.com/content/pdf/10.1007/978-3-031-43901-8.pdf) | -
| 12 | 2023 | ICCV | ELF | ELFNet: Evidential Local-global Fusion for Stereo Matching | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/ICCV2023/papers/Lou_ELFNet_Evidential_Local-global_Fusion_for_Stereo_Matching_ICCV_2023_paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/jimmy19991222/ELFNet)](https://github.com/jimmy19991222/ELFNet) 
| 13 | 2023 | Neural Computing and Applications | - | Region-based evidential deep learning to quantify uncertainty and improve robustness of brain tumor segmentation | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://link.springer.com/content/pdf/10.1007/s00521-022-08016-4.pdf) | -
| 14 | 2023 | ACM MM | - | Learning Discriminative Feature Representation for Open Set Action Recognition | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://dl.acm.org/doi/pdf/10.1145/3581783.3611824) | -
| 15 | 2024 | AAAI | EUMS-3D | Evidential Uncertainty-Guided Mitochondria Segmentation for 3D EM Images | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/download/28287/28565) | -
| 16 | 2024 | TGRS | SSEL | Spectral-Spatial Evidential Learning Network for Open-Set Hyperspectral Image Classification | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10379821) | -

### Natural Language Processing
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|


### Cross-modal Learning
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
| 1 | 2021 | NeurIPS | MoNIG | Trustworthy multimodal regression with mixture of normal-inverse gamma distributions | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.neurips.cc/paper/2021/file/371bce7dc83817b7893bcdeed13799b5-Paper.pdf) | [![GitHub stars](https://img.shields.io/github/stars/MaHuanAAA/MoNIG)](https://github.com/MaHuanAAA/MoNIG) 
| 2 | 2022 | ICRA | USNet | Fast road segmentation via uncertainty-aware symmetric network | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9812452) | [![GitHub stars](https://img.shields.io/github/stars/morancyc/USNet)](https://github.com/morancyc/USNet) 
| 3 | 2022 | International Journal of Approximate Reasoning | DEF | Deep evidential fusion network for medical image classification | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://www.sciencedirect.com/science/article/pii/S0888613X22001256/pdfft?md5=0ed024f031fb6b9593c3ab88bdc917b2&pid=1-s2.0-S0888613X22001256-main.pdf) | -
| 4 | 2022 | ACM MM | DECL | Deep Evidential Learning with Noisy Correspondence for Cross-modal Retrieval | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://dl.acm.org/doi/pdf/10.1145/3503161.3547922) | [![GitHub stars](https://img.shields.io/github/stars/QinYang79/DECL)](https://github.com/QinYang79/DECL) 
| 5 | 2023 | NeurIPS | PAU | Prototype-based Aleatoric Uncertainty Quantification for Cross-modal Retrieval | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.neurips.cc/paper_files/paper/2023/file/4d893f766ab60e5337659b9e71883af4-Paper-Conference.pdf) | [![GitHub stars](https://img.shields.io/github/stars/leolee99/PAU)](https://github.com/leolee99/PAU) 
| 6 | 2023 | ACM MM | DCEL | DCEL: Deep Cross-modal Evidential Learning for Text-Based Person Retrieval | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://dl.acm.org/doi/pdf/10.1145/3581783.3612244) | -
| 7 | 2023 | ICCV | - | Flexible Visual Recognition by Evidential Modeling of Confusion and Ignorance | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/ICCV2023/papers/Fan_Flexible_Visual_Recognition_by_Evidential_Modeling_of_Confusion_and_Ignorance_ICCV_2023_paper.pdf) | -
| 8 | 2023 | arXiv | - | Integrating Large Pre-trained Models into Multimodal Named Entity Recognition with Evidential Fusion | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2306.16991) | -
| 9 | 2024 | Information Fusion | DDEF | Dual-level Deep Evidential Fusion: Integrating multimodal information for enhanced reliable decision-making in deep learning | [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://www.sciencedirect.com/science/article/pii/S1566253523004293/pdfft?md5=090f37e0b04a50009073c2f7e7223900&pid=1-s2.0-S1566253523004293-main.pdf) | - 


### Automatic Driving
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
|  1  | 2019 |IVS|-|Deep, spatially coherent inverse sensor models with uncertainty incorporation using the evidential framework|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=8813826)|-
|  2  | 2021 |ICRA|-|Efficient and robust lidar-based end-to-end navigation|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9561299)|-
|  3  | 2022 |ICRA|-|Robust monocular localization in sparse hd maps leveraging multi-task uncertainty estimation|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9812266)|-
|  4  | 2023 |Arxiv|-|Distil the informative essence of loop detector data set: Is network-level traffic forecasting hungry for more data?|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2310.20366)|-
|  5  | 2023 |ICCV|DELO|DELO: Deep Evidential LiDAR Odometry using Partial Optimal Transport| [![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openaccess.thecvf.com/content/ICCV2023W/UnCV/papers/Ali_DELO_Deep_Evidential_LiDAR_Odometry_Using_Partial_Optimal_Transport_ICCVW_2023_paper.pdf) |-
|  6  | 2023 |T-RO|EVORA|Evora: Deep evidential traversability learning for risk-aware off-road autonomy|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=10606099)|[![GitHub stars](https://img.shields.io/github/stars/mit-acl/mppi_numba)](https://github.com/mit-acl/mppi_numba)
|  7  | 2023 |CoRL|-|nterpretable self-aware neural networks for robust trajectory prediction|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.mlr.press/v205/itkina23a/itkina23a.pdf)|[![GitHub stars](https://img.shields.io/github/stars/sisl/InterpretableSelfAwarePrediction)](https://github.com/sisl/InterpretableSelfAwarePrediction) |
|  8  | 2023 |AAAI |TrEP|Trep: Transformer-based evidential prediction for pedestrian intention with uncertainty|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/download/25463/25235)|[![GitHub stars](https://img.shields.io/github/stars/zzmonlyyou/TrEP)](https://github.com/zzmonlyyou/TrEP)

### EDL in the Open-World
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
|  1  | 2023 |TII|-|Trustworthy fault diagnosis with uncertainty estimation through evidential convolutional neural networks|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=10035450)|-|
|  2 | 2023 |ICLR|AREO|Adaptive Robust Evidential Optimization For Open Set Detection from Imbalanced Data|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://par.nsf.gov/servlets/purl/10425432)|[![GitHub stars](https://img.shields.io/github/stars/ritmininglab/AREO)](https://github.com/ritmininglab/AREO)|
|  3 | 2023 |AAAI|ANEDL|Adaptive Negative Evidential Deep Learning for Open-set Semi-supervised Learning|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/download/29597/31006)|-|
|  4 | 2024 |ICML|MET|Meta Evidential Transformer for Few-Shot Open-Set Recognition|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://openreview.net/pdf?id=CquFGSIU6w)|[![GitHub stars](https://img.shields.io/github/stars/ritmininglab/MET)](https://github.com/ritmininglab/MET)|
|  5 | 2024 |AAAI|EOD|Towards Evidential and Class Separable Open Set Object Detection|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ojs.aaai.org/index.php/AAAI/article/download/28367/28719)|[![GitHub stars](https://img.shields.io/github/stars/roywang021/EOD)](https://github.com/roywang021/EOD)|


### EDL for Science
| ID | Year | Venue |   Abbr   |  Title   |   PDF   |  Code  |
|:--:|:----:|:-----:|:--------:|:--------:|:-------:|:------:|
|  1 | 2021 |ACS Central Science|-|Evidential deep learning for guided molecular property prediction and discovery|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://pubs.acs.org/doi/epdf/10.1021/acscentsci.1c00546)|[![GitHub stars](https://img.shields.io/github/stars/aamini/chemprop)](https://github.com/aamini/chemprop)
|  2 | 2021 |NeurIPS|-|Evaluating Deep Learning Uncertainty Quantification Methods for Neutrino Physics Applications|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](http://bayesiandeeplearning.org/2021/papers/56.pdf)|-|
|  3 | 2022 |ICML|-|Evidential interactive learning for medical image captioning|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://proceedings.mlr.press/v202/zheng23g/zheng23g.pdf)|[![GitHub stars](https://img.shields.io/github/stars/ritmininglab/EIL-MIC)](https://github.com/ritmininglab/EIL-MIC)|
|  4 | 2022 |TBME|FCNN|Personalized blood glucose prediction for type 1 diabetes using evidential deep learning and meta-learning|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9813400)|-|
|  5 | 2022 |npj Digital Medicine|ARISES|Enhancing self-management in type 1 diabetes with wearables and deep learning|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://www.nature.com/articles/s41746-022-00626-5.pdf)|-|
|  6 | 2022 |IoTJ|-|IoMT-enabled real-time blood glucose prediction with deep learning and edge computing|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumber=9681840)|-|
|  7 | 2023 | npj Computational Materials |-| Single-model uncertainty quantification in neural network potentials does not consistently outperform model ensembles|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://www.nature.com/articles/s41524-023-01180-8.pdf)|[![GitHub stars](https://img.shields.io/github/stars/learningmatter-mit/UQ_singleNN)](https://github.com/learningmatter-mit/UQ_singleNN)|
|  8 | 2023 |Arxiv| Eviprompt |Eviprompt: A training-free evidential prompt generation method for segment anything model in medical images|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2311.06400)|-|
|  9 | 2024 |Arxiv|-|Outlier-Detection for Reactive Machine Learned Potential Energy Surfaces|[![PDF](https://img.shields.io/badge/PDF-View-red?style=plastic)](https://arxiv.org/pdf/2402.17686)|-|
## Feedback

If you have any suggestions or find missing papers, please feel free to contact me.

- [e-mail](mailto:chenmengyuan2021@ia.ac.cn)
- [pull request](https://github.com/MengyuanChen21/Awesome-Evidential-Deep-Learning/pulls)

