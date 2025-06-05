![Python Versions](https://img.shields.io/badge/python-3.8+-brightgreen.svg)

# GT-Kla: a geometric-informed and temporal-integrated framework for lysine lactylation sites prediction

# Abstract

Lysine lactylation (Kla) is a recently identified post-translational modification(PTM) that links glycolysis-derived lactate
to chromatin regulation. Identification of Kla sites plays a pivotal role in further elucidating the physiological mechanisms
of lysine. However, traditional experimental approaches for Kla sites detection are costly and inefficient, underscoring the
necessity for computational alternatives. In this paper, we introduce GT-Kla, a deep learning model for Kla sites prediction
that integrates sequence-based biochemical patterns and spatial geometric relationships through a Seq-Geometric Feature
Fusion Module. Additionally, GT-Kla employs a Temporal-Integrated Transformer Augmented Network to guide the
attention mechanism towards biologically important positions in amino acid sequences. Our experiments demonstrate
that GT-Kla outperforms state-of-the-art Kla sites prediction models on both benchmark and custom datasets, while also
showing strong generalization to other PTM sites prediction tasks, such as lysine crotonylation (Kcr) and serine/threonine
phosphorylation. These results highlight the significant improvement in prediction performance achieved by incorporating
distance-based geometric information and temporal attention, compared to models relying solely on sequence and
structural features. We believe GT-Kla will serve as a valuable tool for PTM sites prediction and inspire future research
on related model development.

![figure.png](https://github.com/02cpyw/GT-Kla/blob/main/model.jpg)

# Requirement:

```console
pip install torch=1.13.1+cu116
pip install torch-geometric=2.6.1
pip install torch-scatter=2.1.0+pt113cu116
pip install torch-sparse=0.6.15+pt113cu116
pip install torch-cluster=1.6.0+pt113cu116
```
