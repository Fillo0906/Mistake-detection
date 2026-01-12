# Mistake-detection

This project has been developed by Silberman De Mello Julia, Fusero Filippo, Cafaro Eleonora for the course Data Analysis and Artificial Intelligence (DAAI) 2025/2026. 

# Overview & Motivation
Procedural activity understanding is crucial for developing AI assistance systems (p. 1). Traditional datasets often focus on correct executions, limiting the ability of AI to identify errors (p. 1). This project addresses this gap by implementing and evaluating models for supervised binary classification of errors in cooking videos, where state changes of ingredients make the task challenging (p. 1).It focuses on error recognition in procedural activities, specifically cooking tasks, using egocentric (first-person) video data from the CaptainCook4D benchmark (pp. 1, 3). We explore various deep learning architectures and feature backbones to identify deviations from standard procedures (p. 1).


# Methodology
The project formulates error recognition as a supervised binary classification problem at both sub-step and step levels (p. 2). It leverages transfer learning from models pre-trained on large-scale video recognition tasks (p. 1).

# Architectures
+ V1 (MLP Baseline): Processes each sub-step feature vector independently using a Multi-Layer Perceptron (MLP) without explicit temporal modeling (p. 3).
+ V2 (Transformer Baseline): Introduces temporal reasoning by modeling the entire sequence of sub-step features using a Transformer-based architecture (p. 3).
RNN Baseline: A two-layer vanilla RNN implemented to evaluate explicit sequential modeling (p. 3).

# Feature Backbones
The models use features extracted from the following pre-trained backbones (p. 3):
+ Omnivore: A unified vision backbone trained on images, videos, and 3D data, providing robust spatio-temporal representations (pp. 2-3).
+ SlowFast: A dual-pathway architecture designed to capture both slow semantic information and fast motion cues (pp. 2-3).
+ Perception Encoder (PE): A foundation model trained via a contrastive vision-language objective, used to extract robust, general-purpose embeddings (pp. 2-3).

# External Sources
Checkpoints: https://drive.google.com/drive/folders/19GmWI6jZHK6lK3loRD7JLyv_8k3xW0HA?usp=sharing

1 second pre-extracted features: https://drive.google.com/drive/folders/1G2QSBt8prvlA0tL8SURtCMLWHE8gwZBn?usp=sharing
