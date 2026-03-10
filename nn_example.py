# Re-create the full pipeline Python script in one file


# ===============================
# IMDB Sentiment Classification
# Full Pipeline Script
# ===============================

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch


# ===============================
# 1. Load Data
# ===============================
data = pd.read_csv("IMDB_Dataset.csv")

# Encode labels
data['label'] = data['sentiment'].map({'negative':0, 'positive':1})

# ===============================
# 2. EDA
# ===============================
print("Shape:", data.shape)
print("\\nClass Distribution:")
print(data['sentiment'].value_counts())

print("\\nMissing Values:")
print(data.isnull().sum())

print("\\nDuplicate Reviews:", data.duplicated(subset=['review']).sum())

data['char_count'] = data['review'].apply(len)
data['word_count'] = data['review'].apply(lambda x: len(x.split()))

print("\\nWord Count Stats:")
print(data['word_count'].describe())

# Flag rating leakage patterns
def contains_rating(text):
    return bool(re.search(r"\\b\\d+/10\\b", text))

data['has_rating_pattern'] = data['review'].apply(contains_rating)
print("\\nReviews containing rating patterns:", data['has_rating_pattern'].sum())


# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    data['review'],
    data['label'],
    test_size=0.2,
    stratify=data['label'],
    random_state=42
)


# ===============================
# 4. Baseline Model
# ===============================
baseline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)

print("\\n=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_base))
print("F1:", f1_score(y_test, y_pred_base))
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred_base))


# ===============================
# 5. Neural Network (MLP)
# ===============================
nn_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')),
    ('mlp', MLPClassifier(hidden_layer_sizes=(128,64), early_stopping=True, max_iter=15, random_state=42))
])

nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)

print("\\n=== MLP Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_nn))
print("F1:", f1_score(y_test, y_pred_nn))


# ===============================
# 6. Keras Embedding Model
# ===============================
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=250, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=250, padding='post')

model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=250),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model.fit(X_train_pad, y_train, validation_split=0.2, epochs=5, batch_size=32, callbacks=[early_stop])

loss, acc = model.evaluate(X_test_pad, y_test)
print("\\n=== Deep Learning Results ===")
print("Accuracy:", acc)


# ===============================
# 7. Generate Executive Summary PDF
# ===============================
doc = SimpleDocTemplate("IMDB_Executive_Summary.pdf", pagesize=letter)
elements = []
styles = getSampleStyleSheet()

elements.append(Paragraph("Executive Summary – IMDB Sentiment Classification", styles["Heading1"]))
elements.append(Spacer(1, 0.3 * inch))

summary_text = f'''
Objective: Build sentiment classifier for IMDB reviews.

Dataset: 50,000 balanced reviews.

Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_base):.4f}
MLP Accuracy: {accuracy_score(y_test, y_pred_nn):.4f}
Deep Learning Accuracy: {acc:.4f}

Logistic regression performed strongly, with neural networks showing comparable performance.
Future improvements include removing leakage patterns and testing transformer models.
'''

elements.append(Paragraph(summary_text, styles["BodyText"]))

doc.build(elements)

print("\\nPipeline complete. PDF generated.")


file_path = "/mnt/data/imdb_full_pipeline.py"

with open(file_path, "w") as f:
    f.write(script_content)

file_path