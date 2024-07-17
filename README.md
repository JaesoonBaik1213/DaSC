# __Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels__ 

__*Jae Soon Baik, In Young Yoon, Jun Hoon Kim, and Jun Won Choi*__

Official implementation the paper presented on ECCV 2024 titled "Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels". (Pytorch implementation)


<img src="./overview_DASC.png" width="70%" height="70%" alt="decision_boundary"></img>

# Abstract
Deep neural networks have demonstrated remarkable advancements in various fields using large, well-annotated datasets. However, real-world data often exhibit long-tailed distributions and label noise, significantly degrading generalization performance. Recent studies addressing these issues have focused on noisy sample selection methods that estimate the centroid of each class based on high-confidence samples within each target class. The performance of these methods is limited because they use only the training samples within each class for class centroid estimation, making the quality of centroids susceptible to long-tailed distributions and noisy labels. In this study, we present a robust training framework called Distribution-aware Sample Selection and Contrastive Learning (DaSC). Specifically, DaSC introduces a Distribution-aware Class Centroid Estimation (DaCC) to generate enhanced class centroids. DaCC performs weighted averaging of the features from all samples, with weights determined based on model predictions. Additionally, we propose a confidence-aware contrastive learning strategy to obtain balanced and robust representations. The training samples are categorized into high-confidence and low-confidence samples. Our method then applies Semi-supervised Balanced Contrastive Loss (SBCL) using high-confidence samples, leveraging reliable label information to mitigate class bias. For the low-confidence samples, our method computes Mixup-enhanced Instance Discrimination Loss (MIDL) to improve their representations in a self-supervised manner. Our experimental results on CIFAR and real-world noisy-label datasets demonstrate the superior performance of the proposed DaSC compared to previous approaches.

## Prerequisites:   
- Linux
- Python 3.6.9
- pytorch 1.7.1
- cuda 11.0
- numpy
- tensorboard
- torchvision

## Running code

To train the model(s) and evaluate in the paper, run this command:


```
bash cifar10.sh
```

<!- 
## Citation

```
@inproceedings{baik2022st,
  title={ST-CoNAL: Consistency-Based Acquisition Criterion Using Temporal Self-ensemble for Active Learning},
  author={Baik, Jae Soon and Yoon, In Young and Choi, Jun Won},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={3274--3290},
  year={2022}
}
```
-->

## Note
Our code is based on [SFA](https://github.com/HotanLee/SFA) and [TABASCO](https://github.com/Wakings/TABASCO).
