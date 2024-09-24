# AsthmaSCELNet

**AsthmaSCELNet: A Lightweight Supervised Contrastive Embedding Learning Framework For Asthma Classification Using Lung Sounds**

**Authors: Arka Roy, Udit Satija**

[![Paper Link](https://img.shields.io/badge/Paper%20Link-ISCA%20Archive-blue)](https://www.isca-archive.org/interspeech_2023/roy23_interspeech.html#)
[![Paper Link](https://img.shields.io/badge/Paper%20Link-Research%20Gate-green)](https://www.researchgate.net/publication/371043441_AsthmaSCELNet_A_Lightweight_Supervised_Contrastive_Embedding_Learning_Framework_For_Asthma_Classification_Using_Lung_Sounds)
[![YouTube Link](https://img.shields.io/badge/You%20Tube-ISCA%20Archive-red)](https://www.youtube.com/watch?v=RhKNMkBnm5U&t=176s)


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

```bibtex
@inproceedings{roy23_interspeech,
  author={Arka Roy and Udit Satija},
  title={{AsthmaSCELNet: A Lightweight Supervised Contrastive Embedding Learning Framework for Asthma Classification Using Lung Sounds}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={5431--5435},
  doi={10.21437/Interspeech.2023-428}
}

python AsthmaSCELNet-INTERSPEECH/model/AsthmaSCELNet.py
