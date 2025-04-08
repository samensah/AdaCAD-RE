# Relation Extraction Model

This project implements a relation extraction model that identifies the relationship between entities in text using a language model foundation.

## Project Structure

- `config.py`: Central configuration using argparse
- `model.py`: Relation extraction model implementation
- `dataset.py`: Dataset loading and processing
- `utils.py`: Utility functions like custom collation
- `constant.py`: Constant values like relation descriptions
- `main.py`: Main entry point for running the model

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- Accelerate

### Installation

```bash
pip install torch transformers accelerate
```

### Usage

Run the model with default parameters:

```bash
python main.py
```

Customize configuration:

```bash
python main.py --base_model_name "path/to/your/model" --batch_size 4 --data_path "data/your_dataset.json"
```

View all available options:

```bash
python main.py --help
```

## Configuration Options

The following configuration options are available through command-line arguments:

### Model Arguments
- `--base_model_name`: Path or name of the base language model (default: "../Llama-3.2-1B-Instruct")
- `--max_length`: Maximum number of tokens to generate (default: 30)
- `--temperature`: Temperature for sampling; lower values are more deterministic (default: 0.1)

### Data Arguments
- `--data_path`: Path to the data file (default: "data/refind/test.json")
- `--batch_size`: Batch size for evaluation (default: 2)

### System Arguments
- `--num_workers`: Number of workers for data loading (default: 2)
- `--device`: Device to use for computation (default: "cuda" if available, else "cpu")
- `--seed`: Random seed for reproducibility (default: 42)

### Experiment Tracking
- `--output_dir`: Directory to save outputs (default: "./outputs")
- `--log_level`: Logging level (default: "