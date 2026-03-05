# Spam Email Detection using TensorFlow

## 📌 Project Overview
This project implements a deep learning-based spam email classifier using TensorFlow and LSTM networks. The model classifies emails as Spam or Ham (Not Spam) using Natural Language Processing techniques.

## 🚀 Features
- Text preprocessing (cleaning, stopword removal, punctuation removal)
- Tokenization and sequence padding
- LSTM-based deep learning model
- Model training and evaluation
- Accuracy visualization

## 🛠 Technologies Used
- Python
- TensorFlow / Keras
- Pandas
- NLTK
- Matplotlib
- Scikit-learn

## 📊 Model Architecture
- Embedding Layer (word representation)
- LSTM Layer (sequence learning)
- Dense Layers (classification)
- Sigmoid Output (binary classification)

## 📈 Results
The model achieves high accuracy (~95–97%) on unseen test data.

## ▶️ How to Run

1. Clone the repository:
   git clone https://github.com/your-username/spam-email-detector.git

2. Create virtual environment:
   python -m venv venv

3. Activate environment:
   venv\Scripts\activate  (Windows)
   source venv/bin/activate (Mac/Linux)

4. Install dependencies:
   pip install -r requirements.txt

5. Add dataset inside `data/` folder

6. Run:
   python main.py

## 📚 Learning Outcomes
- NLP preprocessing pipeline
- Handling imbalanced datasets
- LSTM model building in TensorFlow
- Model evaluation and validation

---

Built as part of ML learning journey.