# Harmonizing and aligning M/EEG datasets with covariance-based techniques to enhance predictive regression modeling

This repository contains the code and tools to reproduce the results obtained in the paper [Harmonizing and aligning M/EEG datasets with covariance-based techniques to enhance predictive regression modeling](https://doi.org/10.1101/2023.04.27.538550)

## Abstract

Neuroscience studies face challenges in gathering large datasets, which limits the use of machine learning (ML) approaches. One possible solution is to incorporate additional data from large public datasets; however, data collected in different contexts often exhibit systematic differences called dataset shifts. Various factors, e.g., site, device type, experimental protocol, or social characteristics, can lead to substantial divergence of brain signals that can hinder the success of ML across datasets. In this work, we focus on dataset shifts in recordings of brain activity using MEG and EEG. State-of-the-art predictive approaches on M/EEG signals classically represent the data by covariance matrices. Model-based dataset alignment methods can leverage the geometry of covariance matrices, leading to three steps: recentering, re-scaling, and rotation correction. This work explains theoretically how differences in brain activity, anatomy, or device configuration lead to certain shifts in data covariances. Using controlled simulations, the different alignment methods are evaluated. Their practical relevance is evaluated for brain age prediction on one MEG dataset (Cam-CAN, n=646) and two EEG datasets (TUAB, n=1385; LEMON, n=213). When the target sample included recordings from the same subjects with a different task among the same dataset, paired rotation correction was essential (R2 delta of +0.13 (rest-passive) or +0.17 (rest-smt)). When the target dataset included new subjects and a new task, re-centering led to improved performance (R2 delta of +0.096 for rest-passive, R2 delta of +0.045 for rest-smt). For generalization to an independent dataset sampled from a different population and recorded with a different device, re-centering was necessary to achieve brain age prediction performance close to within domain prediction performance. This study demonstrates that the generalization of M/EEG-based regression models across datasets can be substantially enhanced by applying domain adaptation procedures that can statistically harmonize diverse datasets.

## Citation

```
@article{mellot2023harmonizing,
  title={Harmonizing and aligning M/EEG datasets with covariance-based techniques to enhance predictive regression modeling},
  author={Mellot, Apolline and Collas, Antoine and Rodrigues, Pedro LC and Engemann, Denis and Gramfort, Alexandre},
  journal={Imaging Neuroscience},
  volume={1},
  pages={1--23},
  year={2023},
  publisher={MIT Press One Broadway, 12th Floor, Cambridge, Massachusetts 02142, USA~…}
}
```
