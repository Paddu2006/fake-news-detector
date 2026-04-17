# Fake News Detector
# By Padma Shree
# Project 9 of 25

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 50)
print("📰 FAKE NEWS DETECTOR")
print("=" * 50)

# Step 1: Load datasets
true_path = r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\05_resources\datasets\True.csv"
fake_path = r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\05_resources\datasets\Fake.csv"

try:
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)
    print(f"\n✅ Real news articles: {len(true_df)}")
    print(f"✅ Fake news articles: {len(fake_df)}")
except FileNotFoundError:
    print("\n❌ Dataset not found! Please download from Kaggle.")
    print("https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    exit()

# Step 2: Add labels and combine
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

df = pd.concat([true_df, fake_df], ignore_index=True)
print(f"\n📊 Total articles: {len(df)}")

# Step 3: Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 4: Features and target
X = df['text']  # News article text
y = df['label']  # REAL or FAKE

# Step 5: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📚 Training data: {len(X_train)} articles")
print(f"🧪 Testing data: {len(X_test)} articles")

# Step 6: Convert text to numbers (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"\n🔢 Text converted to {X_train_tfidf.shape[1]} numerical features")

# Step 7: Train the model
model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

print("🤖 Model trained successfully!")

# Step 8: Make predictions
y_pred = model.predict(X_test_tfidf)

# Step 9: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 Model Accuracy: {accuracy * 100:.2f}%")

# Step 10: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Confusion Matrix:")
print(f"True REAL: {cm[0,0]} | False FAKE: {cm[0,1]}")
print(f"False REAL: {cm[1,0]} | True FAKE: {cm[1,1]}")

# Step 11: Classification Report
print(f"\n📋 Detailed Report:")
print(classification_report(y_test, y_pred))

# ============================================
# CHART 1: Confusion Matrix Heatmap
# ============================================
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['REAL', 'FAKE'],
            yticklabels=['REAL', 'FAKE'])
plt.title('Confusion Matrix - Fake News Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("✅ Chart 1 saved: confusion_matrix.png")

# ============================================
# CHART 2: Dataset Distribution (Real vs Fake)
# ============================================
plt.figure(figsize=(6, 6))
label_counts = df['label'].value_counts()
plt.pie(label_counts, labels=['REAL News', 'FAKE News'], 
        autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
plt.title('Dataset Distribution: REAL vs FAKE News')
plt.tight_layout()
plt.savefig('dataset_distribution.png')
plt.show()
print("✅ Chart 2 saved: dataset_distribution.png")

# ============================================
# CHART 3: Sample predictions - First 20 test articles
# ============================================
sample_results = pd.DataFrame({
    'Actual': y_test.values[:20],
    'Predicted': y_pred[:20]
})

plt.figure(figsize=(12, 6))
x_pos = range(len(sample_results))
colors = ['green' if a == p else 'red' for a, p in zip(sample_results['Actual'], sample_results['Predicted'])]
plt.bar(x_pos, [1]*20, color=colors, alpha=0.7)
plt.xlabel('Sample Article')
plt.ylabel('Prediction (REAL=1, FAKE=1)')
plt.title('Sample Predictions: Green = Correct, Red = Wrong')
plt.tight_layout()
plt.savefig('sample_predictions.png')
plt.show()
print("✅ Chart 3 saved: sample_predictions.png")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 50)
print("📋 SUMMARY OF FINDINGS")
print("=" * 50)
print(f"✅ Total Articles Analyzed: {len(df)}")
print(f"✅ REAL News: {len(true_df)}")
print(f"✅ FAKE News: {len(fake_df)}")
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Correct Predictions: {cm[0,0] + cm[1,1]} out of {len(y_test)}")

print("\n" + "=" * 50)
print("✅ PROJECT 9 COMPLETE! 3 charts saved!")
print("=" * 50)