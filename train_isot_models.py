"""
Train AI Models for Fake News Detection
Uses ISOT Dataset (44,898 articles) to train 3 ML models

This script will:
1. Load True.csv and Fake.csv from datasets/ folder
2. Combine title + text for better accuracy
3. Train 3 models: Naive Bayes, Decision Tree, SVM
4. Save trained models to FND_app/ml_models/
5. Show accuracy results (expected: 99.2% ensemble)

Usage:
    python train_isot_models.py

Training time: 5-10 minutes
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import re
import string
import warnings
warnings.filterwarnings('ignore')


class ISOTModelTrainer:
    """Train models on ISOT Fake News Dataset"""
    
    def __init__(self, models_dir='FND_app/ml_models'):
        self.models_dir = models_dir
        self.vectorizer = None
        self.models = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'\@\w+|\#', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def load_isot_dataset(self):
        """Load and combine ISOT dataset (True.csv and Fake.csv)"""
        print("Loading ISOT dataset...")
        
        # Load True news
        true_df = pd.read_csv('datasets/True.csv')
        true_df['label'] = 0  # 0 for REAL
        print(f"✓ Loaded {len(true_df)} REAL news articles")
        
        # Load Fake news
        fake_df = pd.read_csv('datasets/Fake.csv')
        fake_df['label'] = 1  # 1 for FAKE
        print(f"✓ Loaded {len(fake_df)} FAKE news articles")
        
        # Combine datasets
        df = pd.concat([true_df, fake_df], ignore_index=True)
        
        # Combine title and text for better feature extraction
        print("\nCombining title and text...")
        df['content'] = df['title'] + ' ' + df['text']
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total articles: {len(df)}")
        print(f"   REAL news: {len(df[df['label']==0])}")
        print(f"   FAKE news: {len(df[df['label']==1])}")
        
        return df['content'].values, df['label'].values
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train all three models with optimized parameters"""
        print("\n" + "="*60)
        print("TRAINING ML MODELS")
        print("="*60)
        
        results = {}
        
        # 1. Naive Bayes (Fast and effective for text)
        print("\n1️⃣  Training Naive Bayes...")
        nb_model = MultinomialNB(alpha=0.1)
        nb_model.fit(X_train, y_train)
        nb_pred = nb_model.predict(X_test)
        nb_accuracy = accuracy_score(y_test, nb_pred)
        nb_f1 = f1_score(y_test, nb_pred)
        print(f"   ✓ Accuracy: {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)")
        print(f"   ✓ F1-Score: {nb_f1:.4f}")
        self.models['naive_bayes'] = nb_model
        results['naive_bayes'] = {'accuracy': nb_accuracy, 'f1': nb_f1}
        
        # 2. Decision Tree (Optimized depth)
        print("\n2️⃣  Training Decision Tree...")
        dt_model = DecisionTreeClassifier(
            max_depth=50,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        dt_model.fit(X_train, y_train)
        dt_pred = dt_model.predict(X_test)
        dt_accuracy = accuracy_score(y_test, dt_pred)
        dt_f1 = f1_score(y_test, dt_pred)
        print(f"   ✓ Accuracy: {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")
        print(f"   ✓ F1-Score: {dt_f1:.4f}")
        self.models['decision_tree'] = dt_model
        results['decision_tree'] = {'accuracy': dt_accuracy, 'f1': dt_f1}
        
        # 3. SVM (Linear kernel for large datasets)
        print("\n3️⃣  Training SVM (this may take a few minutes)...")
        svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        svm_f1 = f1_score(y_test, svm_pred)
        print(f"   ✓ Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
        print(f"   ✓ F1-Score: {svm_f1:.4f}")
        self.models['svm'] = svm_model
        results['svm'] = {'accuracy': svm_accuracy, 'f1': svm_f1}
        
        # 4. Ensemble prediction (majority voting)
        print("\n4️⃣  Ensemble (Majority Voting)...")
        ensemble_pred = []
        for i in range(X_test.shape[0]):
            votes = [nb_pred[i], dt_pred[i], svm_pred[i]]
            ensemble_pred.append(1 if sum(votes) >= 2 else 0)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_f1 = f1_score(y_test, ensemble_pred)
        print(f"   ✓ Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
        print(f"   ✓ F1-Score: {ensemble_f1:.4f}")
        results['ensemble'] = {'accuracy': ensemble_accuracy, 'f1': ensemble_f1}
        
        # Detailed report
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT (Ensemble)")
        print("="*60)
        print(classification_report(y_test, ensemble_pred, 
                                   target_names=['REAL', 'FAKE']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, ensemble_pred)
        print(f"   True REAL predicted as REAL: {cm[0][0]}")
        print(f"   True REAL predicted as FAKE: {cm[0][1]}")
        print(f"   True FAKE predicted as REAL: {cm[1][0]}")
        print(f"   True FAKE predicted as FAKE: {cm[1][1]}")
        
        return results
    
    def save_models(self):
        """Save trained models and vectorizer"""
        print("\n💾 Saving models...")
        
        # Save vectorizer
        vectorizer_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"✓ Vectorizer saved: {vectorizer_path}")
        
        # Save each model
        for model_name, model in self.models.items():
            model_path = os.path.join(self.models_dir, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            print(f"✓ {model_name.replace('_', ' ').title()} saved: {model_path}")
        
        # Save model info
        info_path = os.path.join(self.models_dir, 'model_info.txt')
        with open(info_path, 'w') as f:
            f.write("ISOT Fake News Detection Models\n")
            f.write("="*50 + "\n")
            f.write(f"Dataset: 44,898 articles\n")
            f.write(f"Training Date: {pd.Timestamp.now()}\n")
            f.write(f"Models: Naive Bayes, Decision Tree, SVM\n")
            f.write(f"Ensemble Method: Majority Voting\n")
        print(f"✓ Model info saved: {info_path}")
        
        print("\n✅ All models saved successfully!")
    
    def train(self, sample_size=None):
        """Main training pipeline"""
        print("\n" + "="*60)
        print("🚀 ISOT FAKE NEWS DETECTION - MODEL TRAINING")
        print("="*60)
        
        # Load data
        texts, labels = self.load_isot_dataset()
        
        # Optional: Use sample for faster training
        if sample_size:
            print(f"\n⚠️  Using sample of {sample_size} articles for faster training")
            indices = np.random.choice(len(texts), sample_size, replace=False)
            texts = texts[indices]
            labels = labels[indices]
        
        # Preprocess
        print("\n🔄 Preprocessing texts...")
        texts = [self.preprocess_text(text) for text in texts]
        print("✓ Preprocessing complete")
        
        # Split data
        print("\n📊 Splitting dataset (80% train, 20% test)...")
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"   Training samples: {len(X_train_text)}")
        print(f"   Testing samples: {len(X_test_text)}")
        
        # Vectorization
        print("\n🔢 Vectorizing text using TF-IDF...")
        print("   (This may take a few minutes for large dataset)")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased for better accuracy
            ngram_range=(1, 2),
            stop_words='english',
            min_df=5,
            max_df=0.8
        )
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)
        print(f"✓ Feature dimensions: {X_train.shape[1]}")
        
        # Train models
        results = self.train_models(X_train, X_test, y_train, y_test)
        
        # Save models
        self.save_models()
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📈 Final Model Accuracies:")
        for model_name, metrics in results.items():
            print(f"   {model_name.replace('_', ' ').title()}: {metrics['accuracy']:.2%}")
        
        print("\n🎯 Next Steps:")
        print("   1. Start Django server: python manage.py runserver")
        print("   2. Test the predictions on the dashboard")
        print("   3. Models will automatically load for predictions")
        print("\n")
        
        return results


def main():
    """Main execution"""
    import sys
    
    print("\n" + "="*60)
    print("ISOT FAKE NEWS DATASET TRAINING")
    print("="*60)
    print("\nOptions:")
    print("1. Full dataset (44,898 articles) - Takes 10-15 minutes")
    print("2. Sample dataset (5,000 articles) - Takes 2-3 minutes")
    print("3. Quick test (1,000 articles) - Takes 30 seconds")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1/2/3) [default: 2]: ").strip() or "2"
    
    trainer = ISOTModelTrainer()
    
    if choice == "1":
        print("\n🔥 Training with FULL dataset (this will take time but best accuracy)")
        trainer.train()
    elif choice == "2":
        print("\n⚡ Training with SAMPLE dataset (good balance of speed and accuracy)")
        trainer.train(sample_size=5000)
    elif choice == "3":
        print("\n🚀 Quick training for testing")
        trainer.train(sample_size=1000)
    else:
        print(f"\n❌ Invalid choice: {choice}")
        return
    
    print("="*60)


if __name__ == '__main__':
    main()
