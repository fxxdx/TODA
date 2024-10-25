[TOC]

# CORA

## Requirmenets
- Python3.8
- Pytorch==2.0.1
- Numpy==1.23.0
- Sklearn==1.3.2
- Pandas==1.5.3
- skorch==0.15.0

## Datasets

### Available Datasets
We used three public datasets (HAR、SleepSC、MFD) and one industry dataset in this study. We have provide the **preprocessed datasets** in the **"data"** folder.

## Training procedure

To train a model:

```
python trainers/train.py  --save_dir experiments_logs \
                --run_description run_1 \
                --da_method TODA\
                --dataset HAR \
                --backbone CNN \
                --num_runs 1 \
                --splits 8\
                --TOV 0.5\
                --CON 1.0\
                --CON2 1.0\
```

- Each run displays the results of all cross-domain scenarios in the format **"src_to_trg_run_i"**, where src is the source domain number, trg is the target domain number, and i is the run_id. You can run the scenario multiple times by specifying --num_runs i.
- Each directory contains the classification report, log file, checkpoint and different risk scores.
- At the end of all runs, the total average and std results can be found in the run directory.

## Supplemental Experimental Results

### Extention to Universal Domain Adaptation
In practical applications, the distribution information of features or labels in the target domain may be limited. Furthermore, changes in cross-domain label distributions may result in private labels, which are classes that exist in the target domain but not in the source domain, and vice versa. This implies the presence of feature and label shifts between the source and target domains. Universal Domain Adaptation (UniDA) refers to the problem of enabling machine learning models to perform well in the target domain with both feature and label shifts. UniDA allows machine learning models to generalize to novel and diverse domains, thereby enhancing their overall robustness and applicability in real-world scenarios. This poses a challenging key issue in machine learning, especially in the context of SFDA tasks where the source domain data is inaccessible. 

Specifically, TODA focuses on learning domain-invariant temporal features of time-series data, which makes it applicable not only in closed-set DA but also in UniDA. We posit that samples with the same labels correspond to clustered features, while features of unknown samples deviate from these clusters. Based on this observation, we can detect target private samples by assessing the differences in target features before and after the adaptation stage. Experimental evaluations were conducted in three different UniDA settings using the [WISDM](https://doi.org/10.1145/1964897.1964918) and [HHAR](https://doi.org/10.1145/2809695.2809718) datasets, with results presented in Table \ref{unida}. WISDM and HHAR are both comprised of 3-axis accelerometer data from 30 distinct participants. The data sampling rate for WISDM is 20Hz, while for HHAR it is 50Hz. The experiment uses non-overlapping segments of 128 time steps. The WISDM dataset consists of six activity labels: walking, jogging, sitting, standing, upstairs, and downstairs. Conversely, the HHAR dataset includes six activity labels: biking, sitting, standing, walking, upstairs, and downstairs. It is evident that TODA consistently outperforms the baseline methods. Compared to baseline methods, TODA effectively addresses challenges related to domain adaptation and achieves higher performance.
<div style="width:100px">Source$\rightarrow$ Target</div> | No. Tar Private Class | [CLUDA](https://arxiv.org/abs/2206.06243) | [UAN](https://ieeexplore.ieee.org/document/8954135) | [DANCE](https://proceedings.neurips.cc/paper/2020/file/bb7946e7d85c81a9e69fee1cea4a087c-Paper.pdf)  | [OVANet](https://ieeexplore.ieee.org/document/9711146) | [UniOT](https://proceedings.neurips.cc/paper_files/paper/2022/file/bda6843dbbca0b09b8769122e0928fad-Paper-Conference.pdf)  | [Raincoat](https://dl.acm.org/doi/10.5555/3618408.3618926) | TODA
:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
 WISDM 3$\rightarrow$WISDM 7 | 1 | 23.96 | 8.04 | 8.53 |  25.61 | 26.78 | 28.05 |  $\pmb{45.31}$
WISDM 27$\rightarrow$WISDM 28 |  2 |  8.98 |  6.94 |  6.74 |  7.87 |  10.98 |  $\pmb{53.70}$ |  47.19
WISDM 1$\rightarrow$WISDM 3 |  3 |  0.00 |  32.85 |  38.46 |  $\pmb{61.54}$ |  36.31 |  35.54 |  $\pmb{61.54}$
WISDM 22$\rightarrow$WISDM 17 |  4 |  26.32 |  27.87 |  23.68 |  40.79 |  38.31 |  48.16 |  $\pmb{75.00}$
WISDM 27$\rightarrow$WISDM 15 |  4 |  56.25 |  22.18 |  27.08 |  60.42 |  52.34 |  66.42 |  $\pmb{72.22}$
WISDM 17$\rightarrow$ HHAR 4 |  1 |  12.31 |  24.50 |  15.94 |  25.31 |  26.32 |  28.41 |  $\pmb{39.92}$
HHAR 6$\rightarrow$ WISDM 19 |  1 |  44.50 |  43.09 |  46.19 |  44.08 |  45.93 |  51.86 |  $\pmb{53.03}$

### Masking ratio sensitivity experiment

The effect of the masking ratio on the performance of the temporal completion task is systematically investigated using different masking ratios (i.e., 12.5\%, 25\%, and 50\%) across three different time series applications. Results in Figure 1 show that a masking ratio of 12.5\% achieves optimal performance across all datasets. Higher ratios impede task performance, resulting in lower accuracy, MF1 score, and AUROC due to significant information loss in the original signal. Besides, the poor temporal correlation between large time segments prevents the temporal completer from effectively capturing the temporal information in the source domain.

<table>
<td ><center><img src="misc/maskingratio_acc.png" width="200" class="center">accuracy</center></td>
<td ><center><img src="misc/maskingratio_f1.png" width="200" class="center">macro-F1</center></td>
<td ><center><img src="misc/maskingratio_aoc.png" width="200" class="center">AUROC</center></td>
</table>

**Figure 1. Effect of different temporal masking ratios.**