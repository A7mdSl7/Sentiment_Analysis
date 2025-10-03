# Twitter Sentiment Analysis - Complete NLP Pipeline & API

A complete sentiment analysis project that processes tweets through the entire NLP pipeline:  
from **data preprocessing and model training** to **deployment via a REST API with Flask**.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pipeline Stages](#pipeline-stages)
- [Model Performance](#model-performance)
- [API Usage](#api-usage)
- [Examples](#examples)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 📖 Overview
This project implements an end-to-end **NLP pipeline** for sentiment analysis on Twitter data (Sentiment140).  
It covers:
1. **Data Preprocessing**: Cleaning, tokenization, lemmatization  
2. **Feature Engineering**: Bag-of-Words, TF-IDF  
3. **Model Training**: Logistic Regression, Naive Bayes, Linear SVM  
4. **Evaluation**: Accuracy, precision, recall, F1-score  
5. **Deployment**: Flask REST API for real-time predictions  

The best model achieves **73.6% accuracy** using **Logistic Regression + TF-IDF**.

---

## ✨ Features
- Comprehensive text preprocessing (URLs, mentions, hashtags, emojis, stopwords, lemmatization with spaCy).  
- Multiple feature extraction methods (**BoW**, **TF-IDF**).  
- Comparison between classic ML models (LogReg, Naive Bayes, SVM).  
- Deployable as a **Flask API** with a simple web GUI.  

---

## 📊 Dataset
- **Sentiment140 Twitter Dataset**: 1.6M tweets labeled positive/negative.  
- In this project: **20K sample** used for training and evaluation.  
- Classes: **Binary classification** → positive / negative.  

---

## 📂 Project Structure


```
sentiment-analysis-app/
│
├── app.py                             # Flask API with GUI
├── README.md                          # Documentation
├── sentiment_model.pkl                # Trained model pipeline
├── requirements.txt                   # Dependencies
├── sentiment_model.pkl                # API testing script
└── Twitter Sentiment Analysis.ipynb   # Full notebook workflow
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/A7mdSl7/sentiment-analysis-app.git
cd sentiment-analysis-app

```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements.txt
```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
spacy
emoji
flask
joblib
python -m spacy download en_core_web_sm
```

### Step 4: Download spaCy Language Model
```
python -m spacy download en_core_web_sm
```

---

## Pipeline Stages

### 1. Exploratory Data Analysis (EDA)
- **Class Distribution**: Balanced dataset (50% positive, 50% negative)
- **Tweet Length**: Average ~100 characters
- **Common Tokens**: Analyzed most frequent words per sentiment

### 2. Text Preprocessing

**Regex Cleaning:**
```python
- Remove URLs: http://example.com → ""
- Remove mentions: @user → ""
- Remove hashtags: #topic → ""
- Replace emojis: 😊 → <EMOJI>
- Remove extra whitespace
```

**Linguistic Processing:**
```python
- Case folding: "AMAZING" → "amazing"
- Tokenization: "I love it" → ["I", "love", "it"]
- Stopword removal: ["I", "love", "it"] → ["love"]
- Lemmatization: "running" → "run", "better" → "good"
```

### 3. Feature Engineering

**Text Representations:**
- **Bag-of-Words**: Word count vectors (5000 features)
- **TF-IDF**: Weighted term frequency (5000 features)

**POS Features:**
- Noun count per tweet
- Verb count per tweet
- Adjective count per tweet

### 4. Model Training & Comparison

| Model               | Vectorizer | Accuracy | F1-Score |
|---------------------|------------|----------|----------|
| Logistic Regression | TF-IDF     | **73.6%**| **0.74** |
| Logistic Regression | BoW        | 73.2%    | 0.73     |
| Naive Bayes         | BoW        | 72.6%    | 0.73     |
| Naive Bayes         | TF-IDF     | 72.3%    | 0.72     |
| Linear SVM          | TF-IDF     | 71.5%    | 0.71     |
| Linear SVM          | BoW        | 70.6%    | 0.71     |

**Best Model**: Logistic Regression with TF-IDF

### 5. Model Evaluation

**Classification Report (Best Model):**
```
              precision    recall  f1-score   support

    negative       0.75      0.71      0.73      1987
    positive       0.73      0.77      0.75      2013

    accuracy                           0.74      4000
   macro avg       0.74      0.74      0.74      4000
weighted avg       0.74      0.74      0.74      4000
```

**Error Analysis:**
- Analyzed misclassified tweets
- Common errors: sarcasm, mixed sentiment, context-dependent phrases

---

## API Usage

### Start the API Server
```bash
python app.py
```
Server runs at: `http://localhost:5000`

### API Endpoints

#### 1. Home / Documentation
```bash
GET http://localhost:5000/
```

#### 2. Single Tweet Prediction
```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "tweet": "I love this product!"
}
```

**Response:**
```json
{
  "original_tweet": "I love this product!",
  "cleaned_tweet": "love product",
  "sentiment": "positive"
}
```

#### 3. Batch Prediction
```bash
POST http://localhost:5000/predict/batch
Content-Type: application/json

{
  "tweets": [
    "Great service!",
    "Terrible experience",
    "Highly recommend"
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "original_tweet": "Great service!",
      "cleaned_tweet": "great service",
      "sentiment": "positive"
    },
    ...
  ],
  "count": 3
}
```

---

## Examples

### Example 1: Positive Tweet
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"tweet": "This phone is amazing! Best purchase ever! 📱"}'
```
**Output:** `sentiment: "positive"`

### Example 2: Negative Tweet
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"tweet": "Worst customer service. Never buying again. 😡"}'
```
**Output:** `sentiment: "negative"`

### Example 3: Tweet with Noise
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"tweet": "@company Your new #product is fantastic! https://example.com"}'
```
**Output:** `sentiment: "positive"`
(Mentions, hashtags, and URLs are automatically removed)

### Example 4: Python Client
```python
import requests

url = "http://localhost:5000/predict"
data = {"tweet": "I absolutely love this!"}

response = requests.post(url, json=data)
print(response.json())
```

---

## Model Details

### Architecture
```
Pipeline:
  1. TfidfVectorizer (max_features=5000)
  2. LogisticRegression (max_iter=200)
```

### Preprocessing Steps
1. Regex cleaning (URLs, mentions, hashtags)
2. Text normalization (lowercase, remove non-alphabetic)
3. spaCy NLP processing
4. Lemmatization
5. Stopword removal

### Hyperparameters
- **TF-IDF**: max_features=5000, default parameters
- **Logistic Regression**: max_iter=200, default solver (lbfgs)

---

## Future Improvements

### Model Enhancements
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add multi-class sentiment (positive, negative, neutral)
- [ ] Incorporate Word2Vec/GloVe embeddings
- [ ] Handle sarcasm and irony detection

### API Enhancements
- [ ] Add authentication (API keys or JWT)
- [ ] Implement rate limiting
- [ ] Add confidence scores to predictions
- [ ] Create Swagger/OpenAPI documentation
- [ ] Add caching for repeated queries

### Deployment
- [ ] Containerize with Docker
- [ ] Deploy to cloud (AWS, GCP, Azure)
- [ ] Set up CI/CD pipeline
- [ ] Add monitoring and logging
- [ ] Create frontend web interface

### Data & Evaluation
- [ ] Expand dataset with recent tweets
- [ ] Cross-validation for robust evaluation
- [ ] A/B testing for model improvements
- [ ] Real-time feedback loop

---

## Technical Stack

- **Language**: Python 3.8+
- **ML Framework**: scikit-learn
- **NLP Library**: spaCy, NLTK
- **Web Framework**: Flask
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

---

## Troubleshooting

### Common Issues

**1. Model file not found**
```
Error: FileNotFoundError: sentiment_model.pkl
Solution: Ensure sentiment_model.pkl is in the same directory as app.py
```

**2. spaCy model not found**
```
Error: Can't find model 'en_core_web_sm'
Solution: Run: python -m spacy download en_core_web_sm
```

**3. Port already in use**
```
Error: Address already in use
Solution: Change port in app.py: app.run(port=5001)
```

---

## Acknowledgments

- **Dataset**: Sentiment140 by Stanford University
- **Libraries**: scikit-learn, spaCy, Flask, NLTK
- **Inspiration**: NLP course labs and real-world sentiment analysis applications

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For questions or feedback:
- Open an issue on GitHub
- Submit a pull request
- Contact: [eng.ahmedsala7ali@gmail.com]

---

**Note**: This project is for educational purposes. For production use, consider additional security measures, scalability improvements, and comprehensive testing.
