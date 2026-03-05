"""
Spam Email Detection
Step 1: Imports and environment check
"""

# Data handling
import pandas as pd
import numpy as np
import tensorflow as tf


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# NLP utilities
import nltk
from nltk.corpus import stopwords

# ML utilities (we won't use them yet)
from sklearn.model_selection import train_test_split

print("All libraries imported successfully!")

# =========================
# STEP 3: LOAD DATASET
# =========================

# Load the CSV file into a DataFrame
data = pd.read_csv("data/Emails.csv")

# Basic sanity checks
print("Dataset loaded successfully")
print("Shape of dataset:", data.shape)

# View first 5 rows
print(data.head())

# =========================
# STEP 4: UNDERSTAND LABELS
# =========================

# Check unique labels
print("Unique labels:", data["label"].unique())

# Count number of spam and ham emails
label_counts = data["label"].value_counts()
print("\nLabel counts:")
print(label_counts)

# =========================
# STEP 5: VISUALIZE LABEL DISTRIBUTION
# =========================

plt.figure(figsize=(6, 4))
sns.countplot(x="label", data=data)

plt.title("Spam vs Ham Distribution")
plt.xlabel("Email Type")
plt.ylabel("Count")

plt.show()

# =========================
# STEP 6: BALANCE THE DATASET
# =========================

# Separate ham and spam emails
ham_emails = data[data["label"] == "ham"]
spam_emails = data[data["label"] == "spam"]

print("Before balancing:")
print("Ham count:", len(ham_emails))
print("Spam count:", len(spam_emails))

# Downsample ham emails to match spam count
ham_downsampled = ham_emails.sample(
    n=len(spam_emails),
    random_state=42
)

# Combine balanced dataset
balanced_data = pd.concat([ham_downsampled, spam_emails])

print("\nAfter balancing:")
print(balanced_data["label"].value_counts())

# =========================
# STEP 7: TEXT CLEANING
# =========================

# Remove the word "Subject" from email text
balanced_data["text"] = balanced_data["text"].str.replace("Subject", "")

# Check result
print(balanced_data["text"].head())

# -------------------------
# STEP 7.2: REMOVE PUNCTUATION
# -------------------------

import string

# List of all punctuation characters
punctuations = string.punctuation

def remove_punctuation(text):
    """
    Removes punctuation from a given text string
    """
    translator = str.maketrans("", "", punctuations)
    return text.translate(translator)

# Apply punctuation removal
balanced_data["text"] = balanced_data["text"].apply(remove_punctuation)

# Check result
print(balanced_data["text"].head())

# -------------------------
# STEP 7.3: REMOVE STOPWORDS
# -------------------------

# Download stopwords list (runs once, cached after)
nltk.download("stopwords")

def remove_stopwords(text):
    """
    Removes English stopwords from text
    """
    stop_words = set(stopwords.words("english"))
    cleaned_words = []

    for word in text.split():
        word = word.lower()
        if word not in stop_words:
            cleaned_words.append(word)

    return " ".join(cleaned_words)

# Apply stopword removal
balanced_data["text"] = balanced_data["text"].apply(remove_stopwords)

# Check cleaned text
print(balanced_data["text"].head())

# =========================
# STEP 8: TOKENIZATION
# =========================

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Split features and labels
X = balanced_data["text"]
y = balanced_data["label"]

# Convert labels to binary (spam=1, ham=0)
y = (y == "spam").astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -------------------------
# STEP 8.2: CREATE TOKENIZER
# -------------------------

# Create tokenizer object
tokenizer = Tokenizer()

# Learn word-to-index mapping from training data
tokenizer.fit_on_texts(X_train)

# Total number of unique words
vocab_size = len(tokenizer.word_index) + 1

print("Vocabulary size:", vocab_size)

# -------------------------
# STEP 8.3: TEXT TO SEQUENCES
# -------------------------

# Convert training text to sequences
X_train_sequences = tokenizer.texts_to_sequences(X_train)

# Convert testing text to sequences
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Inspect one example
print("Original text:")
print(X_train.iloc[0])

print("\nTokenized sequence:")
print(X_train_sequences[0])


# -------------------------
# STEP 8.4: PAD SEQUENCES
# -------------------------

# Define maximum sequence length
max_len = 100

# Pad training sequences
X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=max_len,
    padding="post",
    truncating="post"
)

# Pad testing sequences
X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=max_len,
    padding="post",
    truncating="post"
)

# Inspect padded result
print("Padded sequence shape:", X_train_padded.shape)
print("Example padded sequence:")
print(X_train_padded[0])

# =========================
# STEP 9: BUILD THE MODEL
# =========================

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=32,
        input_length=max_len
    ),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# =========================
# STEP 10: COMPILE THE MODEL
# =========================

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
# =========================
# VERIFY MODEL STRUCTURE
# =========================

# Force model to build using input shape
model.build(input_shape=(None, max_len))

# Now print summary again
model.summary()


# =========================
# STEP 11: TRAIN THE MODEL
# =========================

history = model.fit(
    X_train_padded,
    y_train,
    validation_data=(X_test_padded, y_test),
    epochs=10,
    batch_size=32
)


# =========================
# STEP 12: EVALUATE MODEL
# =========================

test_loss, test_accuracy = model.evaluate(
    X_test_padded,
    y_test
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# =========================
# STEP 12.2: VISUALIZE TRAINING
# =========================

plt.figure(figsize=(6, 4))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
