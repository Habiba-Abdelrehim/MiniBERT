# MiniBERT: Lightweight Emotion Classification Model

A compact, BERT-inspired language model for **emotion classification**, fine-tuned on the [GoEmotions dataset](https://huggingface.co/datasets/go_emotions) using **Hugging Face** and **PyTorch**.

---

## Features
- **Lightweight BERT-Inspired Architecture**:  
  Custom-built transformer model with significantly fewer parameters than standard BERT, designed for speed and efficiency.
  
- **Dataset Preprocessing**:  
  GoEmotions dataset processed to handle single-label emotion classification by filtering out multi-label samples.

- **Training Pipeline**:  
  Fine-tuned using GPU and SLURM-based training for scalable compute environments.

---

## How It Works

1. **Dataset Preprocessing**:
   - Filters out multi-label samples from GoEmotions.
   - Tokenizes text using Hugging Face tokenizers.
   - Vectorizes text using `CountVectorizer` for baseline comparison.

2. **Custom Model**:
   - Compact BERT-style transformer with reduced parameter count (~18M).
   - Embedding layers, transformer blocks, and classification layers designed from scratch.

3. **Training and Evaluation**:
   - Trained for multi-class emotion classification using PyTorch.
   - Current results show the model tends to predict a dominant class, indicating the need for improved handling of class imbalance.
   - Evaluation includes accuracy, per-class precision, recall, F1-scores, and confusion matrices.

4. **Visualization**:
   - Plots for confusion matrices and evaluation metrics to analyze model performance and class distribution.

---

## Current Results

| Metric              | Value      |
|---------------------|------------|
| Test Accuracy       | ~35%       |
| Weighted Precision  | 0.12       |
| Weighted Recall     | 0.35       |
| Weighted F1 Score   | 0.18       |

> ⚠️ **Note:** The current model performance is limited, with strong bias toward a dominant class. Future improvements will focus on handling class imbalance, adjusting model complexity, and optimizing hyperparameters.

---

## Future Work
- Incorporate class weighting or focal loss to mitigate class imbalance.
- Experiment with deeper architectures or additional transformer layers.
- Extend training schedules and conduct broader hyperparameter sweeps.
- Compare directly to Hugging Face’s pretrained BERT-base for benchmarking.

---

## Tools and Frameworks
- PyTorch  
- Hugging Face Datasets and Tokenizers  
- Google Colab (GPU support)  
- SLURM-based training pipeline (for external cluster runs)

---

## References
- [GoEmotions Dataset](https://huggingface.co/datasets/go_emotions)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

---

## Acknowledgements
This project was developed as part of an academic research exploration in lightweight large language models and emotion classification.
