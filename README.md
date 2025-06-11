# AsthmaSCELNet

**AsthmaSCELNet: A Lightweight Supervised Contrastive Embedding Learning Framework For Asthma Classification Using Lung Sounds**

**Authors: Arka Roy, Udit Satija**

[![Paper Link](https://img.shields.io/badge/Paper%20Link-ISCA%20Archive-blue)](https://www.isca-archive.org/interspeech_2023/roy23_interspeech.html#)
[![Paper Link](https://img.shields.io/badge/Paper%20Link-Research%20Gate-green)](https://www.researchgate.net/publication/371043441_AsthmaSCELNet_A_Lightweight_Supervised_Contrastive_Embedding_Learning_Framework_For_Asthma_Classification_Using_Lung_Sounds)
[![YouTube Link](https://img.shields.io/badge/YouTube-IML%20Reading-red)](https://www.youtube.com/watch?v=RhKNMkBnm5U&t=176s)


# Abstract
<p align="justify">
Asthma is one of the most prevalent respiratory disorders, which can be identified by different modalities such as speech, wheezing of lung sounds (LSs), spirometric measures, etc. In this paper, we propose AsthmaSCELNet, a lightweight supervised contrastive embedding learning framework, to classify asthmatic LSs by providing adequate classification margin across the embeddings of healthy and asthma LS, in contrast to vanilla supervised learning. Our proposed framework consists of three steps: pre-processing, melspectrogram extraction, and classification. The AsthmaSCELNet consists of two stages: embedding learning using a lightweight embedding extraction backbone module that extracts compact embedding from the melspectrogram, and classification by the learnt embeddings using multi-layer perceptrons. The proposed framework achieves an accuracy, sensitivity, and specificity of 98.54%, 98.27%, and 98.73% respectively, that outperforms existing methods based on LSs and other modalities.</p>

# Methodology 
![20](https://github.com/user-attachments/assets/f35131c5-553d-4534-a7ff-7ca4651b643b)

# Dataset
**Dataset Link:** 
[![Paper Link](https://img.shields.io/badge/KAUH%20Data-Mendeley%20Data-yellow)](https://data.mendeley.com/datasets/jwyy9np4gv/3)
![image](https://github.com/user-attachments/assets/47f1325e-4459-4278-a0f6-76b548cec49c)

# Results
![image](https://github.com/user-attachments/assets/5266069c-e2d2-4bd4-a080-7abc56cfe5ae)

# Performance 
![image](https://github.com/user-attachments/assets/6aafc958-aee8-42eb-a848-b6b174c6b1b6)


# Cite as:
A. Roy, U. Satija, "AsthmaSCELNet: A Lightweight Supervised Contrastive Embedding Learning Framework for Asthma Classification Using Lung Sounds", in *Proc. INTERSPEECH 2023*, 5431-5435, doi: 10.21437/Interspeech.2023-428.

**Also cite the related works**

A. Pal, A. Roy and U. Satija, "A Unified Joint Contrastive Triplet Loss with Temporal and Frequency Signal Fusion for Diagnosing Heart Murmurs," ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Hyderabad, India, 2025, pp. 1-5, doi: 10.1109/ICASSP49660.2025.10889389.



```bibtex
@inproceedings{roy23_interspeech,
  author={Arka Roy and Udit Satija},
  title={{AsthmaSCELNet: A Lightweight Supervised Contrastive Embedding Learning Framework for Asthma Classification Using Lung Sounds}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={5431--5435},
  doi={10.21437/Interspeech.2023-428}
}
@INPROCEEDINGS{10889389,
  author={Pal, Ayushi and Roy, Arka and Satija, Udit},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={A Unified Joint Contrastive Triplet Loss with Temporal and Frequency Signal Fusion for Diagnosing Heart Murmurs}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Heart;Support vector machines;Time-frequency analysis;Accuracy;Databases;Contrastive learning;Signal processing;Nearest neighbor methods;Speech processing;Phonocardiography;Cardiovascular disorder (CVD);heart murmurs (HM);phonocardiogram (PCG);supervised contrastive learning based triplet network},
  doi={10.1109/ICASSP49660.2025.10889389}}



