# BOLT ⚡️: Bash Optimization through Lightweight Transformers

[BOLT Leaderboard](https://bolt-dashboard.replit.app/)  

Efficient large language models that convert natural language descriptions into shell/bash commands, using optimization techniques such as pruning, quantization, assisted decoding and fine-tuning. This repository contains the code for training, evaluating and visualizing the results with a web leaderboard. Make your bash commands faster, smaller and more efficient! 🚀

## 🌟 Features

- **Quantization**: Quantize the model weights to reduce the model size and improve inference speed
- **Pruning**: Prune the model to reduce the model size
- **Assisted Decoding**: Use an ensemble of language models to improve the quality of the generated commands
- **Fine-tuning**: Fine-tune the model on a custom dataset to improve the model's performance on specific tasks
- **Command Validation**: Built-in syntax validation for generated bash commands
- **Evaluation System**: Using LLM-as-a-judge to evaluate the quality of the generated commands
- **Resource Monitoring**: CPU and memory usage tracking during model operations
- **Web Leaderboard**: Interactive interface for testing and visualizing model outputs

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Hugging Face Transformers

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yanismiraoui/cs229s-project.git
cd cs229s-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Usage

### Evaluation 🧪

To evaluate the model, run the following command after setting the parameters in the `configs/config.yaml` file:

```bash
python evaluate.py
```

Note: To use the LLM-as-a-judge, you need to have an OpenAI API key and save it in a `OPENAI_API_KEY` file in the root directory.

### Web Leaderboard 🏆

To launch the web interface:

```bash
python dashboard/app.py
```

## 🔧 Project Structure

```
.
├── configs/            # Configuration files
├── dashboard/         # Web interface
├── evaluators/        # Evaluation modules
├── models/           # Model architecture
├── utils/            # Utility functions
│   ├── finetune.py
│   ├── evaluation_utils.py
│   └── command_validator.py
└── evaluate.py       # Evaluation script
```

## 📊 Evaluation Metrics

The system uses multiple evaluation metrics:
- Levenshtein distance for command similarity
- Semantic similarity using model embeddings
- Command syntax validation
- LLM-as-a-judge for command quality evaluation using the GPT-4o model and NL2BASH dataset
- Resource usage monitoring (CPU, memory and wall-clock time)
