# GT-Kla
GT-Kla: a geometric-informed and temporal-integrated framework for lysine lactylation sites prediction

Lysine lactylation (Kla) is a recently identified post-translational modification (PTM) that links glycolysis-derived lactate to chromatin regulation. Identification of Kla sites plays a pivotal role in further elucidating the physiological mechanisms of lysine. However, traditional experiment approaches for Kla site detection are costly and inefficient, underscoring the need for computational alternatives. In this paper, we introduce GT-Kla, a deep learning model for Kla site prediction that integrates sequence-based biochemical patterns and spatial geometric relationships through a Seq-Geometric Feature Fusion Module (SGFF). Additionally, GT-Kla employs a Temporal-Integrated Transformer Attention Network (TITAN) to refine temporal dependencies in residue dynamics. Our experiments demonstrate that GT-Kla outperforms state-of-the-art Kla site prediction models on both benchmark and custom datasets, while also showing strong generalization to other PTM site prediction tasks, such as lysine crotonylation (Kcr) and serine/threonine phosphorylation. These results highlight the significant improvement in prediction performance achieved by incorporating distance-based geometric information and temporal attention, compared to models relying solely on sequence and structural features. We believe GT-Kla will serve as a valuable tool for Kla site prediction and contribute to the development of more efficient models in future research.

1、We first propose GT-Kla, a deep learning model with
a temporal-weighted attention mechanism that integrates
sequence and distance-based geometric information, offering
a novel approach for the prediction of lysine lactylation
sites.
2、Our model outperforms the current outstanding Kla
site prediction models on both constructed and publicly
available Kla datasets, which demonstrates GT-Kla’s strong
robustness.
3、Performance in Kcr and S/T phosphorylation site prediction
tasks highlights the remarkable generalization ability of GT-
Kla, offering a solution approach for other types of PTM site
prediction tasks.
