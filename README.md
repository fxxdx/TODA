# TODA

## Rebuttal

### To reviewer #1 Comparasion with other time-series SFDA methods
#### HAR dataset
| scenario | TemSR Acc | TemSR MF1 | E-MAPU acc | E-MAPU MF1 |
|----------|-----------|-----------|------------|------------|
| 2->11    | 0.6316    | 0.5432    | 100        | 100        |
| 6->23    | 0.3214    | 0.1965    | 97.98      | 97.82      |
| 7->13    | 0.4343    | 0.38      | 98.01      | 98.59      |
| 9->18    | 0.7182    | 0.6766    | 88.94      | 88.37      |
| 12->16   | 0.4727    | 0.4894    | 80.45      | 72.85      |
| 3->10    | 0.2584    | 0.2217    | 62.67      | 62.35      |
| 11->9    | 0.3218    | 0.2387    | 60.85      | 57.49      |
| 5->10    | 0.3371    | 0.2851    | 65.93      | 65.69      |
| 7->10    | 0.3258    | 0.2785    | 48.31      | 47.8       |
| 10->16   | 0.6       | 0.5582    | 77.67      | 78.42      |
| 5->9     | 0.4713    | 0.4306    | 68.15      | 67.92      |

#### SleepSC dataset
| scenario | TemSR Acc | TemSR MF1 | E-MAPU acc  | E-MAPU MF1   |
|----------|-----------|-----------|-------------|--------------|
| 0->11    | 0.5734    | 0.4661    | 58.43       | 51.24        |
| 12->5    | 0.7725    | 0.6713    | 77.39       | 64.88        |
| 7->18    | 0.7468    | 0.7198    | 73.46       | 70.59        |
| 16->1    | 0.7176    | 0.5806    | 71.69       | 62.22        |
| 9->14    | 0.8773    | 0.715     | 90.38       | 73.66        |
| 2->11    | 0.5092    | 0.4919    | 47.66476667 | 40.24866667  |
| 9->5     | 0.7452    | 0.6024    | 76.4699     | 64.09195533  |
| 12->11   | 0.6093    | 0.5367    | 60.60566667 | 50.62783333  |
| 17->11   | 0.555     | 0.4427    | 63.20616667 | 57.58913333  |
| 16->11   | 0.6243    | 0.5029    | 47.48433333 | 37.57133333  |
| 6->9     | 0.8155    | 0.6609    | 82.3155     | 71.72295667  |
#### MFD
| scenario | TemSR Acc | TemSR MF1 | E-MAPU acc  | E-MAPU MF1   |
|----------|-----------|-----------|-------------|--------------|
| 0->1     | 0.9945    | 0.9959    | 98.62       | 98.83        |
| 1->2     | 0.7259    | 0.7755    | 89.45       | 92.05        |
| 3->1     | 0.9933    | 0.9951    | 99.04       | 99.95        |
| 1->0     | 0.8013    | 0.8259    | 87.19       | 89.43        |
| 2->3     | 0.9046    | 0.9292    | 97.95       | 99.89        |
| 1->3     | 0.9989    | 0.9992    | 99.89       | 99.87        |
| 0->2     | 0.879     | 0.9106    | 75.50833333 | 83.621       |
| 0->3     | 0.9978    | 0.9984    | 96.92333333 | 96.97147     |
| 2->0     | 0.7758    | 0.7399    | 76.78333333 | 66.29333333  |
| 2->1     | 0.9046    | 0.9293    | 97.05533333 | 97.68833333  |
| 3->0     | 0.7958    | 0.8206    | 87.01666667 | 86.37966667  |

### To reviewer #4 Significance t-tests on TODA with the other methods
The results denotes the T-test of TODA with each method in the first line of the table.
#### Acc on SleepSC dataset
||MAPU|Rainboat|DIRT|NRC|AaD|SHOT|CORAL|VRADA|Codats|DANN|
|-|-|--|-|-|-|-|-|-|-|-|
|0->11|0.0036|0.0005|0.0016|0.0013|0.0294|0.0195|0.0011|0.0407|0.0276|0.0008|
|12->5|0.0448|0.0002|0.0025|0.0001|0.0142|0.0296|0.0002|0.0323|0.0149|0.0009|
|7->18|0.0244|0.0001|0.0001|0.0003|0.0032|0.0542|0.0016|0.0441|0.0289|0.0019|
|16->1|0.0397|0.0009|0.0001|0.0016|0.0481|0.0054|0.0002|0.0191|0.0233|0.0031|
|9->14|0.0182|0.0001|0.0001|0.0111|0.0339|0.0084|0.0025|0.0218|0.0276|0.0002|
|2->11|0.0247|0.0001|0.0101|0.0441|0.0405|0.0530|0.0059|0.0117|0.0341|0.0082|
|9->5|0.0457|0.0003|0.0001|0.0041|0.0313|0.0017|0.0005|0.0040|0.0417|0.0218|
|12->11|0.0318|0.0062|0.0111|0.0229|0.0177|0.0030|0.0044|0.0429|0.0433|0.0152|
|17->11|0.0004|0.0001|0.0009|0.0491|0.0137|0.0088|0.0003|0.0308|0.0364|0.0200|
|16->11|0.0176|0.0034|0.0021|0.0200|0.0149|0.0093|0.0013|0.0488|0.0510|0.0547|
|6->9|0.0108|0.0001|0.0001|0.0034|0.0157|0.0110|0.0001|0.0283|0.0103|0.0003|

### To reviewer #4 Comparison Experiment on WISDM dataset (Acc %, MF1 %)
|        | R-DANN Acc | R-DANN MF1 | VRADA Acc | VRADA MF1 | CoDATS Acc | CoDATS MF1 | DIRT Acc | DIRT MF1 | CORAL Acc | CORAL MF1 | Raincoat Acc | Raincoat MF1 | SHOT Acc | SHOT MF1 | NRC Acc | NRC MF1 | AaD Acc | AaD MF1 | MAPU Acc | MAPU MF1 | TODA Acc | TODA MF1 |
|--------|------------|------------|-----------|-----------|------------|------------|----------|----------|-----------|-----------|--------------|--------------|----------|----------|---------|---------|---------|---------|----------|----------|----------|----------|
| 3->7   | 7.46       | 8.82       | 7.01      | 7.36      | 41.67      | 9.80       | 9.38     | 6.82     | 9.38      | 6.82      | 28.05        | 12.37        | 10.42    | 4.02     | 37.50   | 25.25   | 38.54   | 26.11   | 42.17    | 24.72    | 45.31    | 27.45    |
| 27->28 | 13.97      | 11.52      | 12.74     | 9.78      | 39.33      | 21.33      | 12.36    | 15.62    | 41.57     | 22.22     | 53.70        | 30.72        | 39.33    | 17.37    | 7.87    | 6.88    | 7.87    | 7.28    | 35.96    | 14.65    | 47.19    | 28.41    |
| 1->3   | 15.64      | 14.91      | 14.96     | 13.49     | 53.85      | 14.00      | 19.23    | 14.80    | 19.23     | 14.80     | 35.54        | 17.43        | 61.54    | 27.43    | 53.85   | 16.97   | 53.85   | 14.14   | 61.54    | 27.43    | 61.54    | 45.71    |
| 22->17 | 19.67      | 15.67      | 16.83     | 14.17     | 82.89      | 46.20      | 67.11    | 38.10    | 73.68     | 41.83     | 48.16        | 32.44        | 42.11    | 19.63    | 43.42   | 23.40   | 43.42   | 23.40   | 70.63    | 39.19    | 75.00    | 48.36    |
| 27->15 | 24.58      | 20.74      | 22.98     | 18.93     | 72.92      | 45.63      | 72.92    | 43.45    | 62.50     | 28.71     | 66.42        | 52.92        | 66.67    | 29.62    | 31.25   | 37.28   | 25.00   | 34.72   | 66.67    | 29.62    | 72.22    | 60.58    |


### To reviewer #5 Apply the Target Domain Data to the Pre-trained Source Encoder (No Adaptation Stage)
#### HAR dataset
|   Source->Target | Acc |MF1|AUC-ROC|
| -----------| -----------| -----------| -----------|
|2_to_11|	0.9474|	0.9439	|1.0000|
|6_to_23	|0.8393	|0.8192	|0.9982|
|7_to_13	|0.9394	|0.9333	|0.9865|
|9_to_18	|0.7818	|0.7635|	0.935|
|12_to_16	|0.7091	|0.7452	|0.916|
|3_to_10	|0.5393	|0.5761|	0.784|
|11_to_9|	0.5287	|0.4662|	0.8071|
|5_to_10|	0.5169	|0.499|	0.769|
|7_to_10	|0.5393|	0.5321	|0.749|
|10_to_16|	0.7909	|0.7667	|0.9725|
|5_to_9|	0.5402|	0.4697|	0.7621|
|Average|	0.6975|	0.6832	|0.8799|


#### SleepSC dataset
|   Source->Target | Acc |MF1|AUC-ROC|
| -----------| -----------| -----------| -----------|
| 0_to_11 |0.5800|  0.4893| 0.8154|
|12_to_5|0.6719|0.4889|0.8786|
|7_to_18|0.7208|0.6567|0.9242|
|16_to_1|0.6317|0.5541|0.9005|
|9_to_14|0.7836|0.6332|0.9033|
|2_to_11|0.4432|0.4053|0.7890|
|9_to_5|0.6406|0.4837|0.8717|
|12_to_11|0.3276|0.2575|0.6570|
|17_to_11|0.4277|0.3769|0.6913|
|16_to_11|0.3430|0.2911|0.6860|
|6_to_9|0.8274|0.7150|0.9269|
|Average|0.5816|0.4865|0.8222|

#### MFD dataset
|   Source->Target | Acc |MF1|AUC-ROC|
| -----------| -----------| -----------| -----------|
|0_to_1	|0.7125|	0.5864|	0.8363
|1_to_2	|0.8779|	0.9006	|0.9654
|3_to_1	|1.000	|1.000|	1.000|
|1_to_0	|0.8202|	0.7871|	0.96
|2_to_3	|0.9789	|0.9845	|0.9996
|1_to_3	|1.000|	1.000	|1.000|
|0_to_2|0.6482	|0.4712|	0.8366
|0_to_3	|0.7248|	0.6556|	0.8113
|2_to_0|	0.8058|	0.7738	|0.8911
|2_to_1|	0.9667|	0.9755	|0.9996
|3_to_0|	0.7492|	0.7332|	0.9471
|Average	|0.844	|0.8062|	0.9315


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

Specifically, TODA focuses on learning domain-invariant temporal features of time-series data, which makes it applicable not only in closed-set DA but also in UniDA. We posit that samples with the same labels correspond to clustered features, while features of unknown samples deviate from these clusters. Based on this observation, we can detect target private samples by assessing the differences in target features before and after the adaptation stage. Experimental evaluations were conducted in three different UniDA settings using the [WISDM](https://doi.org/10.1145/1964897.1964918) and [HHAR](https://doi.org/10.1145/2809695.2809718) datasets, with results presented in Table 1. WISDM and HHAR are both comprised of 3-axis accelerometer data from 30 distinct participants. The data sampling rate for WISDM is 20Hz, while for HHAR it is 50Hz. The experiment uses non-overlapping segments of 128 time steps. The WISDM dataset consists of six activity labels: walking, jogging, sitting, standing, upstairs, and downstairs. Conversely, the HHAR dataset includes six activity labels: biking, sitting, standing, walking, upstairs, and downstairs. It is evident that TODA consistently outperforms the baseline methods. Compared to baseline methods, TODA effectively addresses challenges related to domain adaptation and achieves higher performance.
<table border="1">
  <caption><strong>Table 1. Accuracy of UniDA using WISDM, WISDM&rarr;HHAR, WISDM&rarr;HHAR. Shown: mean Accuracy over 5 independent runs. Higher accuracy is better. Best value in bold.</strong></caption>
    <tr>
        <td>Source&rarr;Target</div></td>
        <td>No. Tar Private Class </td>
        <td><p><a href="https://arxiv.org/abs/2206.06243">CLUDA</a></p> </td>
        <td><p><a href="https://ieeexplore.ieee.org/document/8954135">UAN</a></p></td>
        <td><p><a href="https://proceedings.neurips.cc/paper/2020/file/bb7946e7d85c81a9e69fee1cea4a087c-Paper.pdf">DANCE</a></p></td>
        <td><p><a href="https://ieeexplore.ieee.org/document/9711146">OVANet</a></p></td>
        <td><p><a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/bda6843dbbca0b09b8769122e0928fad-Paper-Conference.pdf">UniOT</a></p></td>
        <td><p><a href="https://dl.acm.org/doi/10.5555/3618408.3618926">Raincoat</a></p></td>
        <td>TODA</td>
    </tr>
    <tr>
        <td>WISDM 3&rarr;WISDM 7</td>
        <td>1</td>
        <td>23.96</td>
        <td>8.04</td>
        <td>8.53</td>
        <td>25.61</td>
        <td>26.78</td>
        <td>28.05</td>
        <td><b>45.31</b></td>
    </tr>
    <tr>
          <td>WISDM 27&rarr;WISDM 28</td>
          <td>2</td>
          <td>8.98</td>
          <td>6.94</td>
          <td>6.74</td>
          <td>7.87</td>
          <td>10.98</td>
          <td><b>53.70</b></td>
          <td>47.19</td>
      </tr>
     <tr>
          <td>WISDM 1&rarr;WISDM 3</td>
          <td>3</td>
          <td>0.00</td>
          <td>32.85</td>
          <td>38.46</td>
          <td><b>61.54</b></td>
          <td>36.31</td>
          <td>35.54</td>
         <td><b>61.54</b></td>
      </tr>
      <tr>
        <td>WISDM 22&rarr;WISDM 17</td>
        <td>4</td>
        <td>26.32</td>
        <td>27.87</td>
        <td>23.68</td>
        <td>40.79</td>
        <td>38.31</td>
        <td>48.16</td>
        <td><b>75.00</b></td>
    </tr>
      <tr>
        <td>WISDM WISDM 27&rarr;WISDM 15</td>
        <td>4</td>
        <td>56.25</td>
        <td>22.18</td>
        <td>27.08</td>
        <td>60.42</td>
        <td>52.34</td>
        <td>66.42</td>
        <td><b>72.22</b></td>
    </tr>
    <tr>
        <td>WISDM 17&rarr;HHAR 4</td>
        <td>1</td>
        <td>12.31</td>
        <td>24.50</td>
        <td>15.94</td>
        <td>25.31</td>
        <td>26.32</td>
        <td>28.41</td>
        <td><b>39.92</b></td>
    </tr>
      <tr>
        <td>HHAR 6&rarr;WISDM 19</td>
        <td>1</td>
        <td>44.50</td>
        <td>43.09</td>
        <td>46.19</td>
        <td>44.08</td>
        <td>45.93</td>
        <td>51.86</td>
        <td><b>53.03</b></td>
    </tr>
</table>


### Masking ratio sensitivity experiment

The effect of the masking ratio on the performance of the temporal completion task is systematically investigated using different masking ratios (i.e., 12.5\%, 25\%, and 50\%) across three different time series applications. Results in Figure 1 show that a masking ratio of 12.5\% achieves optimal performance across all datasets. Higher ratios impede task performance, resulting in lower accuracy, MF1 score, and AUROC due to significant information loss in the original signal. Besides, the poor temporal correlation between large time segments prevents the temporal completer from effectively capturing the temporal information in the source domain.

<table>
<td ><center><img src="misc/maskingratio_acc.png" width="200" class="center">accuracy</center></td>
<td ><center><img src="misc/maskingratio_f1.png" width="200" class="center">macro-F1</center></td>
<td ><center><img src="misc/maskingratio_aoc.png" width="200" class="center">AUROC</center></td>
</table>

**Figure 1. Effect of different temporal masking ratios.**
