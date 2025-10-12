# Individual vocal recognition of Ovenbird songs

This repository contains code associated with the [preprint](https://www.biorxiv.org/content/10.1101/2025.06.26.661638v1.full.pdf) "Automated acoustic individual identification enables ecological insights from passive monitoring data".

## Citation
```
Lapp, Sam, R. Patrick Lyon, Scott J. Wilson, Tessa A. Rhinehart, Chapin Czarnecki, Lauren Chronister, Cameron J. Fiss, Jeffery L. Larkin, Erin Bayne, and Justin Kitzes, 2025. "Automated identification of individual birds by song enables multi-year recapture from passive acoustic monitoring data." bioRxiv.
```

```bibtex
@article{lapp2025automated,
  title={Automated identification of individual birds by song enables multi-year recapture from passive acoustic monitoring data},
  author={Lapp, Sam and Lyon, R Patrick and Wilson, Scott J and Rhinehart, Tessa A and Czarnecki, Chapin and Chronister, Lauren and Fiss, Cameron J and Larkin, Jeffery L and Bayne, Erin and Kitzes, Justin},
  journal={bioRxiv},
  year={2025},
}
```

## Overview

This codebase implements a complete pipeline for **Automated Acoustic Individual Identification (AIID)** of Ovenbirds using passive acoustic monitoring (PAM) data. The system uses deep learning to identify individual birds from their vocalizations with 96% accuracy, enabling non-invasive tracking of 405 individuals across multiple years for ecological research. We provide code and documentation for the extension of our approach to other species. 

### Key Innovation

Rather than requiring manually labeled individual identities for training, our approach uses **recording location as pseudo-labels**. This allows the system to learn individual-specific vocal features from unlabeled PAM data, making large-scale individual recognition feasible for ecological studies.

### How to use this code base

The Python Notebook `demo.ipynb` demonstrates the use of our custom feature extractor for discriminating individual Ovenbirds by song. It uses a small sample dataset of 100 songs from 10 individual Ovenbirds, which is included in this repository. You can look through the code and outputs of the notebook or run it yourself (see "Run a demo..." below).

The `scripts` folder contains the full set of Python and R scripts and notebooks used for the analyses in the manuscript. Some of these rely on large passive acoustic monitoring datasets that are not provided publicly, but are available from the authors upon reasonable request. Others use the publicly available annotated evaluation dataset and passive acoustic monitoring dataset to reproduce the figures and results from the manuscript. 

The `src` folder contains Python implementations of the machine learning approaches described in the manuscript. 

The `results` and `figures` folders contain tabular and visual outputs from our experiments and analyses. 

The `checkpoints` folder contains the model weights of a machine learning model (ResNet 18 CNN in Pytorch) that we trained for the task of discriminating between the songs of individual Ovenbirds. 


## Methodology

Our approach follows a 6-step workflow:

1. **Species Detection**: Use HawkEars model to detect Ovenbird songs in PAM recordings
2. **Feature Extractor Training**: Train ResNet-based models using location pseudo-labels
3. **Embedding Generation**: Extract neural network features from detected songs
4. **Individual Discovery**: Apply clustering (HDBSCAN) to embeddings to identify individuals
5. **Resighting Histories**: Create detection/non-detection matrices for each individual
6. **Ecological Modeling**: Estimate survival rates using Cormack-Jolly-Seber models

## Repository Structure

```
├── src/                         # Core source code
│   ├── dataset.py               # Data loading and batch sampling
│   ├── model.py                 # Neural network architectures
│   ├── loss.py                  # Custom contrastive loss functions
│   ├── preprocessor.py          # Audio preprocessing pipelines
│   └── evaluation.py            # Clustering and evaluation metrics
├── scripts/                     # Analysis workflows
│   ├── 1_prepare_training_data/ # Data preparation and formatting
│   ├── 2_train_feature_extractors/ # Model training scripts
│   ├── 3_evaluate_performance/  # Model evaluation and testing
│   ├── 4_PAM_resighting/        # Individual discovery from PAM data
│   ├── 5_ecological_modeling/   # Survival analysis in R/JAGS
│   └── 6_results/              # Generate manuscript figures/tables
├── results/                     # Analysis outputs
│   ├── model_performance/       # Clustering accuracy metrics
│   ├── survival_model_parameter_estimates/ # CJS model results
│   └── embedding_visualizations/ # t-SNE plots and cluster quality
├── figures/                     # Manuscript figures
│   ├── parameter_posteriors/    # Survival parameter distributions
│   ├── model_comparisons/       # Architecture performance plots
│   └── clustering_examples/     # Individual clustering visualizations
├── checkpoints/                 # Trained model weights
└── demo.ipynb                   # Demo notebook applying feature extractor to sample data
```

## Run a demo of the individual identification model in Google Colab

You can run the demo.ipynb python notebook or simply inspect the results to see how our custom trained feature extractor clusters individuals in a feature space, and inspect a few songs. 

## Reproducing results from the manuscript

### 1. Environment Setup

For Python scripts and notebooks (training, evaluation, mark recapture case study):
Create a python environment and install dependencies
(this command line code snippet assumes you're using conda for managing python environments)

```bash
conda create -n ovenbirds python=3.11
conda activate ovenbirds
pip install -r requirements.txt
```

For ecological modeling in R:
```r
install.packages(c("jagsUI", "AHMbook", "ggplot2", "dplyr", "patchwork", "rjags", "tidyr"))
```

### 2. Training Feature Extractors

Train AIID models using different strategies:

```bash
# Supervised classification approach
cd scripts/2_train_feature_extractors/
python train.py config_supervised.yml

# Contrastive learning with location pseudo-labels
python train.py config_contrastive.yml

# ArcFace loss for embedding learning
python train.py config_arcface.yml
```

Configuration files specify:
- Model architecture (ResNet18/50, HawkEars transfer learning)
- Training strategy (supervised, contrastive, ArcFace)
- Data preprocessing options
- Hyperparameters and training schedule

### 3. Individual Discovery from PAM Data

Process PAM recordings to discover individuals:

```bash
cd scripts/4_PAM_resighting/
python 7_embed_and_reduce_dims.py
```

This script:
- Loads trained AIID model from `/checkpoints/`
- Embeds all PAM clips through the neural network
- Applies t-SNE dimensionality reduction
- Saves 3D embeddings for visualization and clustering

### 4. Ecological Modeling

Estimate survival rates using Bayesian methods:

```r
# Run hierarchical CJS survival models
setwd("scripts/5_ecological_modeling/")
source("1_nimble_CJS.R")
```

Models include:
- **Full model**: Habitat covariates (canopy cover, slope, basal area)
- **Null model**: Site random effects + year/block fixed effects
- **Base model**: Simple CJS without hierarchical structure

Results saved to `/results/survival_model_parameter_estimates/`

### 5. Performance Evaluation

Evaluate clustering performance on labeled test sets:

```bash
cd scripts/3_evaluate_performance/
python evaluate_models.py
```

Generates metrics including:
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Fowlkes-Mallows Index (FMI)
- Bipartite Hungarian matching accuracy

## Key Results

### Individual Recognition Performance
- **99% accuracy** on unfamiliar individuals in the test set
- Robust across different recording conditions and years
- Best performance with ResNet18 + supervised classification

### Ecological Insights
- **405 unique individuals** identified across 2021-2024
- **0.70 apparent annual survival** rate (95% CI: 0.56-0.81)
- Habitat effects on abundance, but not survival, through canopy cover and slope

### Technical Contributions
- Location-based pseudo-labeling enables training without individual IDs
- Contrastive learning was ineffective
- Clustering pipeline reliably discovers individuals from embeddings
- Integration with ecological modeling provides population-level insights

## Code-Manuscript Connections

| Manuscript Element | Code Implementation |
|-------------------|-------------------|
| Table 1: 96% recognition accuracy | `src/evaluation.py`: `bipartite_hungarian_matching_accuracy()` |
| Figure 1: 405 individuals tracked | `scripts/4_PAM_resighting/7_embed_and_reduce_dims.py` |
| Table 2: 0.70 survival rate | `scripts/5_ecological_modeling/1_nimble_CJS.R` |
| Methods: Location pseudo-labels | `src/loss.py`: `ssl_location_loss()` |
| Methods: ResNet architecture | `src/model.py`: `ContrastiveResnet18`, `Resnet18_Classifier` |
| Methods: HDBSCAN clustering | `src/evaluation.py`: `make_pseudolabels()` |
| Supplement: Preprocessing | `src/preprocessor.py`: `OvenbirdPreprocessor` |

## File Descriptions

### Core Source Files

**`src/dataset.py`**
- `AIIDLocalizedClipDataset`: Loads audio clips with individual/location labels
- `PointCodeSampler`: Creates training batches from multiple recording sites
- Custom samplers ensure balanced representation across locations

**`src/model.py`**
- `ContrastiveResnet18/50`: ResNet backbones with projection heads
- `Resnet18/50_Classifier`: Standard classification architectures
- `HawkEarsOneModel`: Transfer learning from pre-trained bioacoustics model

**`src/loss.py`**
- `ssl_location_loss()`: Contrastive loss using location pseudo-labels
- Pushes apart clips from different locations, pulls together clips from same location
- Core innovation enabling training without individual identity labels

**`src/preprocessor.py`**
- `OvenbirdPreprocessor`: Audio preprocessing optimized for Ovenbird vocalizations
- 2-10 kHz bandpass filtering matches Ovenbird frequency range
- Temporal jittering and normalization for robust training

**`src/evaluation.py`**
- `make_pseudolabels()`: Complete pipeline from embeddings to individual assignments
- `cluster()`: HDBSCAN clustering with UMAP/t-SNE dimensionality reduction
- `evaluate()`: Comprehensive clustering performance metrics

### Key Scripts

**`scripts/2_train_feature_extractors/train.py`**
- Main training script supporting multiple learning strategies
- YAML-configurable architecture and hyperparameter settings
- Outputs trained models to `/checkpoints/` directory

**`scripts/4_PAM_resighting/7_embed_and_reduce_dims.py`**
- Processes PAM dataset through trained AIID models
- Generates t-SNE embeddings for visualization and clustering
- Creates basis for individual discovery and resighting histories

**`scripts/5_ecological_modeling/1_nimble_CJS.R`**
- Hierarchical Cormack-Jolly-Seber survival models in JAGS
- Tests multiple model formulations (full, null, base)
- Incorporates site-level habitat covariates and random effects


## Dependencies

- Python 3.8+
- PyTorch 1.12+
- OpenSoundscape 0.9+
- scikit-learn, pandas, numpy
- UMAP-learn, HDBSCAN
- R 4.0+ with JAGS for survival modeling

## Contact

For questions about the code or methodology, please open an issue in this repository or contact Sam Lapp.

---

This work demonstrates how AI can transform passive acoustic monitoring data into detailed ecological insights, enabling non-invasive individual tracking at unprecedented scales for wildlife research and conservation.