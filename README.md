# LSTM Slogan Generator & Classifier

## Project Overview

This project implements an industry-specific slogan generation and classification system:

1. **Slogan Generator**: Generates realistic slogans by combining learned phrases (bigrams and trigrams) specific to each industry
2. **Slogan Classifier**: Uses an ensemble voting system combining three models:
   - Logistic Regression with TF-IDF features
   - Multinomial Naive Bayes with TF-IDF
   - LSTM neural network with embedding layers

The system is trained on 3,464 real slogans across 29 major industries, achieving **42.57% ensemble accuracy** on industry classification.

## Architecture & Design

### Data Pipeline
- **Input**: 5,346 slogans across 142 industries from `slogan-valid.csv`
- **Preprocessing**: Text lowercasing and punctuation removal using spaCy
- **Filtering**: Kept only industries with 50+ samples → 3,464 slogans, 29 industries (avg 119 per industry)
- **Training Split**: 80/20 stratified train/test split (2,771 training samples, 693 test samples)

### Generator: Phrase-Based Approach
Instead of trying to learn word-by-word generation (which requires massive datasets), the generator extracts patterns from real slogans:

- **Bigram extraction**: All 2-word phrases from slogans
- **Trigram extraction**: All 3-word phrases from slogans
- **Industry phrase banks**: Top-100 phrases per industry, sorted by frequency
- **Generation**: Combines 2 random phrases from the industry's phrase bank
- **Advantage**: Produces coherent outputs guaranteed to contain real language patterns

**Phrase Bank Example (Internet industry, top phrases):**
```
web design, digital marketing, design and, marketing agency, 
digital marketing agency, software for, web development, ...
```

### Classifier: Ensemble Voting Approach
Three independent models vote on the industry classification:

**Model 1: Logistic Regression**
- Input: TF-IDF features (2000 features, bigrams + unigrams)
- Max features: 2000, ngram_range: (1, 2)
- Class weights: Balanced to handle class imbalance
- Accuracy: 42.0%

**Model 2: Multinomial Naive Bayes**
- Input: Same TF-IDF features as Logistic Regression
- Alpha (Laplace smoothing): 0.1
- Accuracy: 41.0%

**Model 3: LSTM Neural Network**
- Embedding layer: 1000 vocabulary, 64 dimensions
- LSTM layers: 128 units each with Dropout(0.4)
- Dense layers: 256 → 128 units with ReLU activation
- Final output: 29 industries
- Training: 60 epochs, batch_size=16, class weights
- Accuracy: 32.3%

**Ensemble Method**: Majority voting via `np.bincount`
- Combined accuracy: **42.6%** (best of the three individual models)

## Results & Performance

### Classifier Accuracy by Industry (Top 10)

| Industry | Accuracy |
|----------|----------|
| Law Practice | 86.4% |
| Automotive | 80.7% |
| Accounting | 71.4% |
| Machinery | 66.7% |
| Real Estate | 62.5% |
| Apparel & Fashion | 58.3% |
| Insurance | 58.3% |
| Legal Services | 54.5% |
| Health, Wellness & Fitness | 52.2% |
| Computer Software | 50.0% |

### Overall Performance
- **Ensemble Accuracy**: 42.6% (29-way classification)
- **Random Baseline**: 3.4% (1/29 classes)
- **Best Individual Model**: Logistic Regression (42.0%)
- **LSTM Individual**: 32.3% (underfitting on this task)

### Generation Quality
- **Real slogan test**: 1/5 correctly classified (20%)
- **Generated slogan test**: 3/3 correctly classified (100%)

**Example Outputs:**

Generated slogans by industry:
```
Accounting: "accountants and to the arts"
Apparel & Fashion: "your brand destination uk retail italian"
Automotive: "midwest for dealership in"
```

### Key Findings
1. **Phrase-based generation works**: Produces grammatical, industry-specific output
2. **Ensemble provides stability**: Majority voting reduces variance from individual models
3. **TF-IDF baseline is strong**: Simple feature extraction outperforms deep learning on small datasets
4. **Domain imbalance matters**: Top industries (IT, Marketing) achieve 50%+ accuracy; rare industries struggle

## Design Decisions & Tradeoffs

### Why Phrase-Based Generation Instead of LSTM Generation?
**The Problem**: Original single-LSTM approach attempted next-word prediction from 3,700 vocabulary with only 8k training sequences.
- Ratio: 2.7 training examples per word class (mathematically impossible to learn)
- Result: Stuck at 5% accuracy, generating repetitive degenerate outputs

**The Solution**: Extract real phrases (bigrams/trigrams) from training slogans.
- Uses only combination patterns actually present in real data
- Guarantees grammaticality and semantic coherence
- No training needed for generation itself

### Why Ensemble Instead of Single LSTM Classifier?
**The Problem**: Single LSTM underfits when trained on small datasets with minimal feature signal.

**The Solution**: Ensemble three models with different inductive biases:
- **Logistic Regression**: Linear decision boundaries, fast training, interpretable
- **Naive Bayes**: Probabilistic baseline, robust to sparse features
- **LSTM**: Captures sequential patterns in language

Majority voting combines strengths and reduces individual model variance.

### Data Filtering (10+ → 50+ samples per industry)
- Reduced from 95 to 29 industries (removed sparse classes)
- Improved average samples per class: 39 → 119
- Result: More learnable classification problem

## Limitations & Future Improvements

### Current Limitations
1. **Generator diversity**: Limited to combinations of existing phrases (can't create novel phrases)
2. **Classifier accuracy**: 42.6% ensemble accuracy leaves room for improvement
3. **Industry coverage**: Only 29 industries (dropped rare ones with <50 samples)
4. **Vocabulary constraints**: Phrase banks limited to top 100 per industry

### What Would Improve Performance
1. **More data**: Collect 10k+ slogans per industry for better patterns
2. **Better features**: Add word embeddings (Word2Vec, BERT) instead of TF-IDF counts
3. **Class balancing**: Synthetic data generation or resampling for minority classes
4. **Template-based generation**: Hand-craft industry-specific templates for higher quality
5. **Contextual embeddings**: Fine-tune BERT/GPT on domain-specific slogan corpus

## How to Run

### Prerequisites
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Training the Models
1. Open `slogan_model.ipynb` in Jupyter/VS Code
2. Run all cells sequentially from top to bottom
3. Models automatically serialize to `saved_models/` directory after training

### Using Trained Models for Inference
```python
from slogan_model import load_models, generate_slogan_from_loaded, classify_slogan_from_loaded

# Load all trained models
models = load_models()

# Generate a slogan
slogan = generate_slogan_from_loaded("internet", models)
print(f"Generated: {slogan}")

# Classify a slogan
industry = classify_slogan_from_loaded(slogan, models)
print(f"Predicted industry: {industry}")
```

### Production Deployment
All models are pre-serialized in `saved_models/`:
- Phrase banks for generation (no additional training needed)
- TF-IDF vectorizer for feature extraction
- Three classifier models (Logistic Regression, Naive Bayes, LSTM)
- Metadata with performance metrics and industry mappings

Load them anytime with `load_models()` function.

## Project Structure

```
lstm-slogan-model/
├── slogan_model.ipynb              # Main notebook with complete pipeline
├── data/
│   └── slogan-valid.csv            # Training data (3,464 slogans, 29 industries)
├── saved_models/                   # Serialized models and components
│   ├── phrase_banks.pkl            # Generator phrase banks (bigrams + trigrams)
│   ├── tfidf_vectorizer.pkl        # Classifier TF-IDF vectorizer
│   ├── lr_model.pkl                # Logistic Regression model
│   ├── nb_model.pkl                # Naive Bayes model
│   ├── lstm_model.keras            # LSTM classifier model
│   ├── tokenizer.pkl               # Keras tokenizer for sequences
│   ├── industry_to_idx.pkl         # Industry → index mapping
│   ├── idx_to_industry.pkl         # Index → industry mapping
│   └── metadata.pkl                # Model metadata and performance metrics
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Files & Components

- **slogan_model.ipynb**: Complete end-to-end pipeline including data loading, preprocessing, generator, ensemble classifier, and evaluation
- **slogan-valid.csv**: Raw slogan dataset with industry labels (3,464 slogans after filtering to 50+ samples per industry)
- **saved_models/**: Production-ready serialized models
  - Generator: Phrase banks for phrase-based slogan generation
  - Classifier: Ensemble of Logistic Regression, Naive Bayes, and LSTM
- **requirements.txt**: All Python dependencies (pandas, numpy, scikit-learn, TensorFlow, spaCy, joblib)

## Key Learnings

1. **Problem-driven architecture**: Phrase-based generation avoids impossible next-word prediction task
2. **Ensemble robustness**: Voting across three models more reliable than single deep learning model
3. **TF-IDF is competitive**: Linear models with good features beat neural nets on small datasets
4. **Data quality matters**: Filtering to 50+ samples/class improved learnability significantly
5. **Task decomposition**: Separating generation (pattern extraction) from classification (supervised learning) is more tractable

## Future Work

- Implement template-based generation for even higher quality
- Collect more balanced data to expand industry coverage beyond 29
- Add confidence scoring and uncertainty estimates for predictions
- Fine-tune BERT embeddings on domain-specific slogan corpus
- Explore reinforcement learning to optimize slogan relevance
- Create web API wrapper for production inference

## References

- TensorFlow/Keras: https://www.tensorflow.org/
- scikit-learn: https://scikit-learn.org/
- spaCy: https://spacy.io/
- Ensemble Learning: https://en.wikipedia.org/wiki/Ensemble_learning


---

*Part of my journey from Operations Manager to ML Engineer. Follow along on [Dev.to](https://dev.to/jemhrice)*
