# AT&T Spam Detector - CDSD Certification Project

Complete spam detection system comparing three approaches:
1. **Baseline**: DummyClassifier + TF-IDF + Logistic Regression
2. **Advanced**: DistilBERT (transfer learning)

## Model Performance Comparison (Test Set)

| Model | Accuracy | F1-Spam | Recall-Spam | Precision-Spam | False Positives | False Negatives |
|-------|----------|---------|-------------|----------------|-----------------|-----------------|
| Dummy (most frequent) | 86.7% | 0.0% | 0.0% | 0.0% | 0 | 112 |
| TF-IDF + Logistic Regression | 98.0% | 93.0% | 91.0% | 96.0% | ~5 | ~10 |
| **DistilBERT (best)** | **99.5%** | **98.2%** | **96.4%** | **100%** | **0** | **4** |

*Test set: 836 messages (724 ham, 112 spam). All models evaluated on same test set.*

## Key Findings

1. **DistilBERT significantly outperforms** traditional methods:
   - +1.5% accuracy vs TF-IDF/LR
   - +5.2% F1-spam vs TF-IDF/LR
   - **Zero false positives** (critical for user experience)

2. **TF-IDF/LR is a strong baseline** (98% accuracy) but:
   - Lower recall (91% vs 96.4%)
   - Some false positives (~5 cases)

3. **DummyClassifier** shows the class imbalance problem:
   - 86.7% accuracy by always predicting "ham"
   - 0% spam detection (unacceptable)

## Why DistilBERT Wins

- **Context understanding**: "FREE" in "not free" vs "FREE prize"
- **Word embeddings**: Captures semantic similarity (e.g., "win" ≈ "prize" ≈ "free")
- **Attention mechanism**: Focuses on spam indicators regardless of position

## Training Details

- **Model**: DistilBERT (66M parameters)
- **Epochs**: 3 (early stopping on validation F1-spam)
- **Learning rate**: 2e-5
- **Batch size**: 16
- **Max sequence length**: 64 tokens
- **Optimizer**: AdamW (default from Trainer)
- **Validation metric**: F1-spam (best epoch: 3, F1=96.83%)

## Project Structure

```
├── att_spam_detector_distilbert.ipynb  # Complete notebook with all 3 models
├── data/
│   └── spam.csv
├── att_spam_detector/                 # Saved model (best checkpoint)
├── outputs/
│   ├── logs/                         # Training logs
│   └── figures/                      # Plots (confusion matrix, etc.)
└── README.md
```

## Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook att_spam_detector_distilbert.ipynb

# Or use the saved model for inference
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("./att_spam_detector")
tokenizer = AutoTokenizer.from_pretrained("./att_spam_detector")

def predict_spam(text, threshold=0.5):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        spam_prob = probs[0, 1].item()
    return "SPAM" if spam_prob > threshold else "HAM", spam_prob

# Example
sms = "FREE! Click here to claim your $1000 prize!"
prediction, confidence = predict_spam(sms)
print(f"{prediction} (confidence: {confidence:.2%})")
```

## Error Analysis

### False Negatives (4 cases - spams missed):
These spams avoid typical spam keywords or use social engineering:
1. Messages with subtle urgency ("Your account needs verification")
2. Messages with obfuscated links ("Check [link] for details")
3. Messages mimicking legitimate services ("Your package delivery failed")

### False Positives (0 cases):
✅ **Perfect precision** - no legitimate messages incorrectly flagged as spam.
This is crucial for user experience (no false alarms).

## Limitations & Future Work

- **Dataset size**: Only 5,574 messages. More data would improve recall.
- **Concept drift**: Spam patterns evolve. Need periodic retraining.
- **Explainability**: Could add LIME/SHAP for model interpretability.
- **Multilingual**: Currently English only.
- **URL analysis**: Could extract and analyze URLs separately.

## License

MIT

## Author

[Dreipfelt] - CDSD Certification Candidate  
[GitHub](https://github.com/Dreipfelt)

## Acknowledgments

- Dataset: [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Model: [DistilBERT](https://huggingface.co/distilbert-base-uncased) by HuggingFace
- Framework: [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
```
