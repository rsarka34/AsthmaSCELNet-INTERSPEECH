# AsthmaSCELNet

**AsthmaSCELNet: A Lightweight Supervised Contrastive Embedding Learning Framework For Asthma Classification Using Lung Sounds**

**Authors: Arka Roy, Udit Satija**

[![Paper Link](https://img.shields.io/badge/Paper%20Link-IEEE%20Xplore-blue)](https://www.isca-archive.org/interspeech_2023/roy23_interspeech.html#)

# Abstract
Asthma is one of the most prevalent respiratory disorders, which can be identified by different modalities such as speech, wheezing of lung sounds (LSs), spirometric measures, etc. In this paper, we propose AsthmaSCELNet, a lightweight supervised contrastive embedding learning framework, to classify asthmatic LSs by providing adequate classification margin across the embeddings of healthy and asthma LS, in contrast to vanilla supervised learning. Our proposed framework consists of three steps: pre-processing, melspectrogram extraction, and classification. The AsthmaSCELNet consists of two stages: embedding learning using a lightweight embedding extraction backbone module that extracts compact embedding from the melspectrogram, and classification by the learnt embeddings using multi-layer perceptrons. The proposed framework achieves an accuracy, sensitivity, and specificity of 98.54%, 98.27%, and 98.73% respectively, that outperforms existing methods based on LSs and other modalities.

# Methodology 
![block_diagram (1)](https://github.com/rsarka34/AsthmaSCELNet/assets/89518952/8b6e47bc-8a4c-45ff-8fc4-05bddaa893ba)

# Dataset
![image](https://github.com/user-attachments/assets/47f1325e-4459-4278-a0f6-76b548cec49c)
**Dataset Link: KAUH** 
[![Paper Link](https://img.shields.io/badge/KAUH%20Data-Mendeley%20Data-blue)](https://data.mendeley.com/datasets/jwyy9np4gv/3)

# Cite as:
A. Roy, U. Satija,"AsthmaSCELNet: A Lightweight Supervised Contrastive Embedding Learning Framework for Asthma Classification Using Lung Sounds", in *Proc. INTERSPEECH 2023*, 5431-5435, doi: 10.21437/Interspeech.2023-428.

```bibtex
@inproceedings{roy23_interspeech,\
  author={Arka Roy and Udit Satija},\
  title={{AsthmaSCELNet: A Lightweight Supervised Contrastive Embedding Learning Framework for Asthma Classification Using Lung Sounds}},\
  year=2023,\
  booktitle={Proc. INTERSPEECH 2023},\
  pages={5431--5435},\
  doi={10.21437/Interspeech.2023-428}}



