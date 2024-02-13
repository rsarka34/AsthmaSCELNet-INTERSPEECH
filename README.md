# AsthmaSCELNet: A Lightweight Supervised Contrastive Embedding Learning Framework For Asthma Classification Using Lung Sounds 
Authors: Arka Roy, Udit Satija, Department of Electrical Engineering, Indian Institute of Technology Patna.\
Published in Proc. INTERSPEECH 2023, Dublin, Ireland.
![block_diagram (1)](https://github.com/rsarka34/AsthmaSCELNet/assets/89518952/8b6e47bc-8a4c-45ff-8fc4-05bddaa893ba)
# Abstract
Asthma is one of the most prevalent respiratory disorders, which can be identified by different modalities such as speech, wheezing of lung sounds (LSs), spirometric measures, etc. In this paper, we propose AsthmaSCELNet, a lightweight supervised contrastive embedding learning framework, to classify asthmatic LSs by providing adequate classification margin across the embeddings of healthy and asthma LS, in contrast to vanilla supervised learning. Our proposed framework consists of three steps: pre-processing, melspectrogram extraction, and classification. The AsthmaSCELNet consists of two stages: embedding learning using a lightweight embedding extraction backbone module that extracts compact embedding from the melspectrogram, and classification by the learnt embeddings using multi-layer perceptrons. The proposed framework achieves an accuracy, sensitivity, and specificity of 98.54%, 98.27%, and 98.73% respectively, that outperforms existing methods based on LSs and other modalities.

![281669068-1fef1052-c748-4371-8008-b9718ad39093](https://github.com/rsarka34/AsthmaSCELNet/assets/89518952/99f37130-00fb-4bcc-9df6-92e8e239c6f5)

Dataset link: https://data.mendeley.com/datasets/jwyy9np4gv/3 \
Paper link: [https://www.isca-speech.org/archive/pdfs/interspeech_2023/roy23_interspeech.pdf](https://www.isca-archive.org/interspeech_2023/roy23_interspeech.html) 

# Cite as:
@inproceedings{roy23_interspeech,\
  author={Arka Roy and Udit Satija},\
  title={{AsthmaSCELNet: A Lightweight Supervised Contrastive Embedding Learning Framework for Asthma Classification Using Lung Sounds}},\
  year=2023,\
  booktitle={Proc. INTERSPEECH 2023},\
  pages={5431--5435},\
  doi={10.21437/Interspeech.2023-428}}



