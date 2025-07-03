# MiniBERT: Optimized Emotion Classification Model

A lightweight language model for **emotion classification**, fine-tuned on the [GoEmotions dataset](https://huggingface.co/datasets/go_emotions) using **HuggingFace** and **PyTorch**.

## Features
- **Custom BERT Architecture**: A smaller, optimized model designed for speed and efficiency.
- **Dataset**: Preprocessed the GoEmotions dataset to handle multi-label emotion classification.
- **Training Pipeline**: Fine-tuned using GPU and SLURM for high-performance training.

## How It Works
1. **Dataset Preprocessing**:
   - Filters multi-label samples.
   - Vectorizes text data using `CountVectorizer`.
2. **Custom Model**:
   - Compact transformer architecture with fewer parameters.
   - Supports embedding, transformer blocks, and classification layers.
3. **Training and Testing**:
   - Trained on a multi-class classification task.
   - Evaluated using accuracy metrics and confusion matrices.
4. **Visualization**: Insightful plots for model evaluation and debugging.
