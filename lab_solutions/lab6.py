# Import Required Libraries
import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from scipy.sparse import hstack

# Function to load data
def load_construction_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the dataset
dataset_path = os.path.abspath(
    os.path.join(script_dir, os.pardir, "datasets", "construction_documents.json")
)

df = load_construction_data(dataset_path)

# Function to process metadata
def process_metadata(df):
    metadata = df[['project_phase', 'author_role']].fillna('Unknown')
    vec = DictVectorizer(sparse=False)
    features = vec.fit_transform(metadata.to_dict(orient='records'))
    return features, vec

metadata_features, metadata_vec = process_metadata(df)

# Function to clean text
def clean_text(text):
    text = re.sub(r'\b(RFI|CO|QA|QC|PR|SI|MM)\b', '', text)
    text = re.sub(r'\d+(\.\d+)?\s*(mm|cm|m|kg|tons?)', '', text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

df['cleaned_content'] = df['content'].apply(clean_text)

# Function to vectorize text
def vectorize_text(text_series):
    vec = TfidfVectorizer()
    features = vec.fit_transform(text_series)
    return features, vec

tfidf_features, tfidf_vec = vectorize_text(df['cleaned_content'])

# Function to combine features
def combine_features(text_features, metadata_features):
    return hstack([text_features, metadata_features])

combined_features = combine_features(tfidf_features, metadata_features)
labels = df['document_type']
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

# Function to train Naive Bayes
def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

model = train_naive_bayes(X_train, y_train)

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()
    return y_pred

y_pred = evaluate_model(model, X_test, y_test)

# Function to analyze misclassifications
def analyze_misclassifications(y_test, y_pred):
    print("\nMisclassifications Analysis:\n")
    y_test_reset = y_test.reset_index(drop=True)  # Ensure alignment
    misclassified = pd.DataFrame({'Actual': y_test_reset, 'Predicted': y_pred})
    mismatches = misclassified[misclassified['Actual'] != misclassified['Predicted']]
    if not mismatches.empty:
        print(mismatches.value_counts().head(10))
    else:
        print("No misclassifications found.")

analyze_misclassifications(y_test, y_pred)

analyze_misclassifications(y_test, y_pred)

# Function to extract top terms (corrected for metadata awareness)
def extract_top_terms(model, tfidf_vec, metadata_vec):
    feature_names = list(tfidf_vec.get_feature_names_out()) + list(metadata_vec.get_feature_names_out())
    for idx, label in enumerate(model.classes_):
        top10 = np.argsort(model.feature_log_prob_[idx])[-10:]
        top_terms = []
        for i in top10:
            if i < len(feature_names):
                top_terms.append(feature_names[i])
            else:
                top_terms.append(f"<metadata feature {i}>")
        print(f"\nTop terms for {label}: {top_terms}")

extract_top_terms(model, tfidf_vec, metadata_vec)

# Function to perform cross-validation
def perform_cross_validation(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"\nCross-Validation Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")

perform_cross_validation(MultinomialNB(), combined_features, labels)

# Compare with Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_test)
print("\nLogistic Regression Classification Report:\n")
print(classification_report(y_test, logistic_pred))

# Simple classification interface
def classify_new_document(text, project_phase, author_role):
    metadata_dict = [{'project_phase': project_phase, 'author_role': author_role}]
    metadata_feature = metadata_vec.transform(metadata_dict)
    cleaned_text = clean_text(text)
    text_feature = tfidf_vec.transform([cleaned_text])
    combined_feature = hstack([text_feature, metadata_feature])
    prediction = model.predict(combined_feature)
    return prediction[0]

sample_text = "We request clarification on the MEP installation procedure."
print("\nClassification of Sample Document:")
print(classify_new_document(sample_text, "Planning", "Project Manager"))

# Analyze temporal patterns
df['date'] = pd.to_datetime(df['date'])
df['year_month'] = df['date'].dt.to_period('M')
df.groupby(['year_month', 'document_type']).size().unstack().plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title("Document Types Over Time")
plt.ylabel("Count")
plt.xlabel("Year-Month")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()