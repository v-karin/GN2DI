
# GN2DI
=======
# GN2DI: Graph Neural Networks for Data Imputation 🌐

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.0%2B-blue)](https://pytorch-geometric.readthedocs.io/)

GN2DI is a scalable Graph Neural Network framework designed for spatial missing data imputation in sensor networks. Our method leverages neural graph construction techniques optimized for the target task, enabling effective handling of both samples and sensors that were unavailable during training.

<p align="center">
  <img src="assets/Figure.png" alt="GN2DI Architecture" width="800"/>
  <br>
  <em>Figure 1: GN2DI Architecture - Illustration of the modeling process and imputation pipeline</em>
</p>

## 🎯 Model Overview

GN2DI operates in two main phases:

1. **Modeling the Sensors with Graph Process**
   - Creates initial weighted adjacency matrices using multiple similarity measures
   - Optimizes graph structure through GNN layers
   - Learns node representations for effective spatial relationship capture

2. **Prediction and Imputation**
   - Utilizes the optimized graph structure 
   - Processes sensor values through specialized GNN layers
   - Generates final predictions for missing values

## 💪 Performance & Baselines

GN2DI has been evaluated against several strong baselines:

### Traditional Methods:
- **MEAN**: Simple mean imputation
- **KNN**: K-Nearest Neighbors imputation
- **IDW**: Inverse Distance Weighting
- **GSSL**: Graph-Based Semi-Supervised Learning

### Advanced Methods:
- **MICE**: Multiple Imputation by Chained Equations
- **AutoENC**: Vanilla autoencoder design
- **N2V**: Combined KNN with MLP using Node2Vec embeddings
- **SSGAN**: Modified version with 1D CNN architecture

### Performance Improvements:

| Dataset | Best Baseline | GN2DI Performance | Improvement |
|---------|--------------|-------------------|-------------|
| METR-LA | AutoENC (MSE: 29.52) | MSE: 27.34 | 7.3% ⬇️ |
| PEMS-BAY | AutoENC (MSE: 5.53) | MSE: 4.73 | 14.5% ⬇️ |

## 🔑 Key Advantages

1. **Superior Spatial Modeling**
   - Leverages multiple similarity measures (Pearson correlation, cosine similarity, RBF kernel)
   - Optimizes graph structure through gradient signals
   - Learns effective node representations

2. **Zero-Shot Capabilities**
   - Can handle new, unseen sensors
   - Maintains performance with network expansion
   - Adaptive to dynamic sensor networks

3. **Fast Fine-Tuning**
   - Quick adaptation to new scenarios (1-2 epochs)
   - Effective for block-missing data patterns
   - Significant improvements after fine-tuning (up to 26% MSE reduction)

## 🚀 Getting Started

### Prerequisites


### Installation

```bash
git clone https://github.com/AmEskandari/GN2DI.git
cd GN2DI
```

### Basic Usage

```python
from gn2di import GN2DI
from gn2di.data import get_dataset

# Load dataset
data = get_dataset("PEMS-BAY")

# Initialize model with optimal parameters
model = GN2DI(
    num_weights=3,              # Number of initial similarity measures
    hidden_dim_pre_weight=32,   # Hidden dimension for weight initialization
    num_lay_pre_weight=4,       # Number of layers in pre-weight module
    in_channel_gl=32,           # Input channels for graph learning
    num_conv_lay_gl=1,          # Number of GNN layers
    hidden_dim_conv_gl=32,      # Hidden dimension for GNN
    hidden_dim_gl=32,           # Hidden dimension for graph learning
    dropout_pre_weight=0.2      # Dropout rate
)

# Train model
model.train(data)

# Impute missing values
imputed_data = model.impute(data)
```

## 📖 Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{eskandari2024gn2di,
  title={GN2DI: A Scalable Graph Neural Network Framework for Spatial Missing Data Imputation in Sensor Networks},
  author={Eskandari, Amir and Jamshidiha, Saeed and Pourahmadi, Vahid},
  booktitle={2024 IEEE International Conference on Future Machine Learning and Data Science (FMLDS)},
  pages={191--196},
  year={2024},
  organization={IEEE}
}
```

## ✉️ Contact

- **Amir Eskandari** - amireskandari@aut.ac.ir
- **Saeed Jamshidiha** - s_jamshidiha@aut.ac.ir
- **Vahid Pourahmadi** - v.pourahmadi@aut.ac.ir

