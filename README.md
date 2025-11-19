# **StrainFish**

`strainfish` is a weighted ensemble machine learning algorithm with multiple DNA sequence encoders and logic, specifically designed for classification of marker sequences.

## Conceived and built by Kranti Konganti, HFP

### Latest version: 0.2.2

- Multiple DNA sequence encoders for **NVIDIA GPU-accelerated** training.
- A weighted Ensemble machine-learning model generation with sensible defaults.
- **NVIDIA GPU-accelerated Learning and Prediction only!**
- **Important Note**: This software is under active development and as such some features are **experimental**. Results should be thoroughly validated and independently verified before use in critical applications or publications.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Training Models](#training-models)
   - [Basic Training Command](#basic-training-command)
   - [Advanced Configuration](#advanced-configuration)
   - [Encoding Methods](#encoding-methods)
4. [Making Predictions](#making-predictions)
   - [Basic Prediction Command](#basic-prediction-command)
   - [Model Management](#model-management)
5. [Configuration Options](#configuration-options)
   - [**XGBoost** Parameters](#xgboost-parameters)
   - [**RandomForest** Parameters](#randomforest-parameters)
   - [**SentencePiece** Parameters](#sentencepiece-parameters)
   - [Imbalance Handling Parameters](#imbalance-handling-parameters)
6. [Test Data and Examples](#test-data-and-examples)
7. [Dependencies](#dependencies)
8. [License](#license)

## Installation

**StrainFish** <u>**requires**</u> Python 3.12 or newer and **NVIDIA GPU** support for its core machine learning processes.

### Step 1: Install **StrainFish**

First, install the base **StrainFish** package from **PyPI**:

```bash
pip install strainfish
```

This command installs **StrainFish** but **not** the necessary `cuML` (GPU) libraries. The package will not be fully functional until `cuML` is installed in Step 2.

### Step 2: Install `cuML`

The following commands ensure that the correct `cuML` version, compatible with your CUDA environment, is installed alongside **StrainFish**. You must choose **one** of the following commands based on your system's CUDA version to install the compatible `cuML` library:

- **For systems with CUDA 12.x:**

    ```bash
    pip install strainfish[cuda-12]
    ```

- **For systems with CUDA 13.x:**

    ```bash
    pip install strainfish[cuda-13]
    ```

### Verify `cuML` and CUDA Installation

After completing Step 2, verify that `cuML` can access the NVIDIA GPU. The following command uses the `nvidia-ml-py` dependency (included with **StrainFish**) to query the driver.

```bash
python -c "import pynvml; pynvml.nvmlInit(); print('\nNVIDIA CUDA driver version:', f'{pynvml.nvmlSystemGetCudaDriverVersion() // 1000}.{(pynvml.nvmlSystemGetCudaDriverVersion() % 100) // 10}'); pynvml.nvmlShutdown();"
```

This should output the CUDA version (e.g., `12.4`) supported by your installed NVIDIA driver. Ensure this matches or is higher than the CUDA version required by your chosen `cuML` package.

## Quick Start

### Training a Model

To train a model on your DNA sequences:

```bash
strainfish train run \
  -f path/to/sequences.fasta \
  -l path/to/labels.csv \
  -o /path/to/models_output_dir/model_prefix
```

### Predicting using a Model

To perform prediction using a trained model:

```bash
strainfish predict run \
  -f path/to/predict_sequences.fasta \
  -m /path/to/models_output_dir/model_prefix \
  -o path/to/results_directory
```

## Training Models

**StrainFish** uses an ensemble approach for both training and prediction (**XGBoost**, **RandomForest** and **NaiveBayes**), and includes multiple DNA sequence encodings, though only one encoding can be used at a time during training.

### Basic Training Command

```bash
strainfish train run \
  -f training_sequences.fasta \                             # Input FASTA file
  -l labels.csv \                                           # Labels CSV (id,label)
  -o /path/to/models_output_dir/model_prefix                # Output directory for models
```

### Advanced Configuration

Optional **StrainFish** configuration options during training:

```bash
strainfish train run \
  -f training_sequences.fasta \
  -l labels.csv \
  -o model_output_dir \
  --encode-method tf \              # Encoding method: sm, sp, or tf
  --kmer 7 \                        # K-mer size for hashing
  --num-hashes 100 \                # Number of hashes per sequence
  --factor 21 \                     # Sequence overlap factor
  --chunk-size 200 \                # Size of DNA chunks
  --pseknc-weight 0.1 \             # Weight for PseKNC encoding
  --xgb-n-estimators 300 \          # XGBoost parameters
  --rf-n-estimators 100 \           # RandomForest parameters
```

### Encoding Methods

**StrainFish** supports three DNA sequence encoding methods:

- **`tf` (TF-IDF)**: TF-IDF vectorization (**Default**)
- **`sp` (SentencePiece)**: Subword tokenization using SentencePiece models (**Experimental**)
- **`sm` (SOMH)**: MinHash based approach with PseKNC and sequencing composition weights (AT/GC ratio) (**Experimental**)

## Making Predictions

### Basic Prediction Command

```bash
strainfish predict run \
  -f prediction_sequences.fasta \                        # Input FASTA file(s)
  -m /path/to/models_output_dir/model_prefix \           # Path to trained model
  -o results_dir                                         # Output directory for predictions
```

### Model Management

List available models:

```bash
strainfish predict list-models
# Or list models stored at a particular models directory:
strainfish predict list-models -md /path/to/models_dir
```

## Configuration Options

Tunable parameters for **StrainFish**.

### **XGBoost** Parameters

View all configurable **XGBoost** parameters:

```bash
strainfish train show-xgb-params
```

Key parameters:

- `--xgb-n-estimators`: Number of boosting rounds
- `--xgb-max-depth`: Maximum tree depth
- `--xgb-learning-rate`: Learning rate for boosting
- `--xgb-subsample`: Subsample ratio of the training instance

### **RandomForest** Parameters

View all configurable **RandomForest** parameters:

```bash
strainfish train show-rf-params
```

Key parameters:

- `--rf-n-estimators`: Number of trees in the forest
- `--rf-max-depth`: Maximum depth of the tree
- `--rf-random-state`: Random seed for reproducibility
- `--rf-min-samples-leaf`: Minimum samples required at a leaf node

### **SentencePiece** Parameters

View all configurable **SentencePiece** parameters:

```bash
strainfish train show-sp-params
```

Key parameters:

- `--sp-vocab-size`: Vocabulary size for tokenization
- `--sp-max-sentence-length`: Maximum sentence length
- `--sp-char-cov`: Character coverage ratio

### Imbalance Handling Parameters

View all imbalance handling parameters:

```bash
strainfish train show-imb-params
```

Key parameters:

- `--imb-smote-k-neighbors`: Number of neighbors for SMOTE
- `--imb-enn-n-neighbors`: Number of neighbors for ENN cleaning

## Test Data and Examples

This repository includes test data in the `tests/test_input/` directory:

- `test.train.fasta`: Training sequences in FASTA format
- `test.train.csv`: Labels file with `id,label` columns
- `predict.fasta`: Sequences for prediction using trained models

You can use these to test **StrainFish** functionality:

```bash
# Train a model using test data
strainfish train run \
  -f tests/test_input/test.train.fasta \
  -l tests/test_input/test.train.csv \
  -o test_output/test_model

# Make predictions on the trained model
strainfish predict run \
  -f tests/test_input/predict.fasta \
  -m test_output/test_model \
  -o prediction_results
```

## Dependencies

**StrainFish** has the following main dependencies:

- **Core ML Libraries (GPU-accelerated)**: numpy, pandas, scikit-learn, xgboost, `cuML` (mandatory for functionality)
- **Sequence Processing**: biopython, sourmash, sentencepiece
- **CLI Interface**: rich, rich-click
- **Utilities**: joblib, psutil, humanize, pynvml
- **Testing**: pytest, pytest-cov

**Note on `cuML`:** The `cuML` library is essential for **StrainFish**'s GPU-accelerated computations. It must be installed explicitly as an extra (e.g., `strainfish[cuml-cu12]`) during the `pip install` command after the base package installation. For a complete and version-specific list of all dependencies, including those for `cuML`, see `pyproject.toml`.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
