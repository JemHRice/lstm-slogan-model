# LSTM Slogan Model 🧠

A deep learning project using Long Short-Term Memory (LSTM) networks to both generate and classify marketing slogans. Built as part of my HyperionDev Data Science Bootcamp.

---

## Overview

Two LSTM models trained on a clean slogan dataset:

1. **Slogan Generator** — given a seed category, generates a new slogan
2. **Slogan Classifier** — takes a generated (or real) slogan and reclassifies it back to its original category

The classifier acting as a validation layer on the generator is the interesting part — if the classifier correctly identifies the category of a generated slogan, the generator has learned something meaningful about how language differs across categories.

---

## Example Output

<!-- ADD: paste 3-5 example generated slogans here with their seed category -->
<!-- Example format:
**Seed Category:** Sports
**Generated Slogan:** "..."
**Classifier Prediction:** Sports ✅
-->

---

## Model Details

### Preprocessing
- Tokenised slogan dataset using Keras Tokenizer
- Converted sentences to padded sequences of uniform length
- One-hot encoded category labels

### Generator (LSTM)
- Architecture: Embedding → LSTM → Dense (softmax)
- Input: seed category
- Output: generated slogan text
<!-- ADD: training accuracy, loss -->

### Classifier (LSTM)
- Architecture: Embedding → LSTM → Dense (softmax)
- Input: slogan text
- Output: predicted category
- **Classifier Accuracy:** <!-- ADD: e.g. 87% -->

---

## Results

<!-- ADD: screenshot of training curves (loss/accuracy) -->
<!-- ADD: confusion matrix if you have one -->

**Classifier Accuracy:** <!-- ADD -->
**Training Loss:** <!-- ADD -->

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/JemHRice/lstm-slogan-model
cd lstm-slogan-model

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook slogan_model.ipynb
```

**Requirements:**
```
tensorflow
keras
pandas
numpy
jupyter
```

---

## Project Structure

```
lstm-slogan-model/
├── slogan_model.ipynb      # Main notebook
├── data/
│   └── slogans.csv         # Clean slogan dataset
├── models/
│   └── generator.h5        # Saved generator model   <!-- ADD if saved -->
│   └── classifier.h5       # Saved classifier model  <!-- ADD if saved -->
├── requirements.txt
└── README.md
```

---

## What I Learned

- How to preprocess and tokenise text data for sequence models
- Padding sequences to uniform length for batch training
- Building LSTM architectures in Keras
- Using one model to validate the output of another

---

*Part of my journey from Operations Manager to ML Engineer. Follow along on [Dev.to](https://dev.to/jemhrice)*
