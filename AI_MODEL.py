import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import gc
import torch
from tqdm import tqdm
import itertools


device = torch.device("mps")

class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=16):
        self.model_name = model_name
        self.batch_size = batch_size
        
    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.model_name)
        if torch.backends.mps.is_available():
            self.model.to(device)
        return self
    
    def transform(self, X):
        batches = []
        for i in tqdm(range(0, len(X), self.batch_size), desc="Encoding text"):
            batch = X.iloc[i:i+self.batch_size].tolist()
            embeddings = self.model.encode(batch, show_progress_bar=False)
            batches.append(embeddings)
            gc.collect()
        return np.vstack(batches)

print("Loading data...")
data = pd.read_csv('train.csv')
toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for label in toxic_labels:
    data[label] = data[label].astype(int)

data['clean'] = (data[toxic_labels].sum(axis=1) == 0).astype(int)

labels = toxic_labels + ['clean']

X = data['comment_text']
y = data[labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

print("Setting up combined feature pipeline...")
combined_features = FeatureUnion([
    ('st', SentenceTransformerVectorizer(model_name="all-MiniLM-L6-v2", batch_size=16)),
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,3)))
])

print("Setting up model pipeline...")
pipeline = Pipeline([
    ('features', combined_features),
    ('svd', TruncatedSVD(n_components=600, random_state=42)),  
    ('clf', OneVsRestClassifier(
         LogisticRegression(
            penalty='l2',
            C=2.5,          
            solver='saga',
            max_iter=1000,
            tol=1e-4,
            random_state=42,
            n_jobs=-1
         )
    ))
])

print("Training model...")
pipeline.fit(X_train, y_train)

def tuned_predict_from_probs(probs, thresholds_dict, labels):
    y_pred = np.zeros(probs.shape)
    for i, label in enumerate(labels):
        y_pred[:, i] = (probs[:, i] >= thresholds_dict[label]).astype(int)
    return y_pred

def compute_subset_accuracy(y_true, y_pred):
    return (y_pred == y_true.to_numpy()).all(axis=1).mean()

print("Computing validation set probabilities...")
val_probs = pipeline.predict_proba(X_val)

print("Tuning thresholds for maximum accuracy on the validation set...")
candidate_thresholds = [0.1, 0.3, 0.5, 0.7]
best_subset_accuracy = 0.0
best_thresholds = {}

for thresholds in itertools.product(candidate_thresholds, repeat=len(labels)):
    current_thresholds = dict(zip(labels, thresholds))
    y_val_pred = tuned_predict_from_probs(val_probs, current_thresholds, labels)
    current_accuracy = compute_subset_accuracy(y_val, y_val_pred)
    if current_accuracy > best_subset_accuracy:
        best_subset_accuracy = current_accuracy
        best_thresholds = current_thresholds.copy()

print("Calculating train set...")
train_probs = pipeline.predict_proba(X_train)
y_train_pred_tuned = tuned_predict_from_probs(train_probs, best_thresholds, labels)
train_subset_accuracy_tuned = compute_subset_accuracy(y_train, y_train_pred_tuned)

print("Calculating test set...")
test_probs = pipeline.predict_proba(X_test)
y_test_pred_tuned = tuned_predict_from_probs(test_probs, best_thresholds, labels)
test_subset_accuracy_tuned = compute_subset_accuracy(y_test, y_test_pred_tuned)

print("Train accuracy:", train_subset_accuracy_tuned)
print("Test accuracy:", test_subset_accuracy_tuned)