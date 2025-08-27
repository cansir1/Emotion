# Emotion Detection from Text (Machine Learning Project)

## Overview
This project predicts the **emotion** expressed in a text sentence using traditional Machine Learning models.  
It uses **TF-IDF** and **Word2Vec** features along with multiple classifiers like Logistic Regression, SVM, XGBoost, and Naive Bayes.  
The dataset contains six emotions: `anger`, `fear`, `joy`, `love`, `sadness`, `surprise`.

---

## Dataset
We used the **[Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)** from Kaggle.  

**Files used:**
- `train.txt` – training data  
- `val.txt` – validation data  
- `test.txt` – test data  

**Note:** Datasets and `kaggle.json` are **not included** in the repo.  
You can download them directly in Google Colab as shown in the code.

---

## Project Pipeline (What I Did)

### 1. **Data Loading**
- Download dataset from Kaggle directly in Colab.
- Load `train.txt`, `val.txt`, and `test.txt` using Pandas.

### 2. **Text Preprocessing**
- Convert text to lowercase.
- Remove punctuation and non-alphabet characters.
- Tokenize and **lemmatize** words.
- Remove English **stopwords**.

### 3. **Feature Extraction**
- **TF-IDF Vectorization** (max_features=5000).
- **Word2Vec Embeddings** using pre-trained Google News (300D) vectors.

### 4. **Model Training & Evaluation**
Trained and evaluated the following models:

- **Logistic Regression** (TF-IDF and Word2Vec)
- **Support Vector Machine (LinearSVC)** (TF-IDF and Word2Vec)
- **XGBoost Classifier** (TF-IDF and Word2Vec)
- **Naive Bayes**  
  - GaussianNB (Word2Vec)
  - MultinomialNB (TF-IDF)

For each model:
- Calculated **accuracy**.
- Generated **classification reports** (precision, recall, F1-score).

### 5. **Hyperparameter Tuning**
- Used **GridSearchCV** for SVM (LinearSVC) to find the best `C` parameter.

### 6. **Final Model**
- Combined **training + validation** data.
- Trained the final **LinearSVC** model with the best hyperparameters (C=1).
- Evaluated it on the **test dataset**.

### 7. **Visualization**
- Plotted the **confusion matrix** on test data using Seaborn heatmap.

---

## Results (Example)
- **Best model:** LinearSVC with TF-IDF features.  
- **Metrics:** Reported accuracy and classification report for test set in the notebook.
- **Confusion Matrix:** Plotted as heatmap.

---
