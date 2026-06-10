"""
Fake News Detection Engine

This module loads trained AI models and makes predictions on news articles.

Models used:
- Naive Bayes: Probabilistic text classifier
- Decision Tree: Rule-based decision maker  
- SVM: Pattern recognition expert

Ensemble voting: All 3 models vote, majority wins!
"""

import os
import joblib
import re
import string
import numpy as np


MIN_WORD_COUNT = 30  # Minimum words needed for a reliable prediction


class FakeNewsPredictor:
    """Predict fake news using ensemble of ML models"""
    
    def __init__(self, models_dir='FND_app/ml_models'):
        self.models_dir = models_dir
        self.vectorizer = None
        self.models = {}
        self.models_loaded = False
        
        # Try to load models
        self.load_models()
    
    def preprocess_text(self, text):
        """Clean and preprocess text data (same as training)"""
        if not text or text.strip() == "":
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
    
    def load_models(self):
        """Load trained models and vectorizer"""
        try:
            # Check if models directory exists
            if not os.path.exists(self.models_dir):
                print(f"Models directory not found: {self.models_dir}")
                return False
            
            # Load vectorizer
            vectorizer_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
            if not os.path.exists(vectorizer_path):
                print(f"Vectorizer not found: {vectorizer_path}")
                return False
            
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Load models
            model_names = ['naive_bayes', 'decision_tree', 'svm']
            for model_name in model_names:
                model_path = os.path.join(self.models_dir, f'{model_name}_model.pkl')
                if not os.path.exists(model_path):
                    print(f"Model not found: {model_path}")
                    return False
                self.models[model_name] = joblib.load(model_path)
            
            self.models_loaded = True
            print("✓ All models loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def predict(self, text):
        """
        Predict if news is fake or real using ensemble voting
        
        Returns:
            dict: {
                'prediction': 'FAKE' or 'REAL',
                'confidence': float (0-100),
                'individual_predictions': {
                    'naive_bayes': 'FAKE'/'REAL',
                    'decision_tree': 'FAKE'/'REAL', 
                    'svm': 'FAKE'/'REAL'
                },
                'vote_count': {'FAKE': int, 'REAL': int}
            }
        """
        if not self.models_loaded:
            return {
                'prediction': 'ERROR',
                'confidence': 0,
                'message': 'Models not trained yet. Please train models first by running: python train_isot_models.py'
            }
        
        if not text or text.strip() == "":
            return {
                'prediction': 'ERROR',
                'confidence': 0,
                'message': 'Please provide news text to analyze'
            }

        # ── NEW: warn if text is too short for reliable prediction ──
        word_count = len(text.split())
        if word_count < MIN_WORD_COUNT:
            return {
                'prediction': 'TOO_SHORT',
                'confidence': 0,
                'message': (
                    f'Your text is only {word_count} word{"s" if word_count != 1 else ""}. '
                    f'Please paste at least {MIN_WORD_COUNT} words (a full paragraph or more) '
                    f'for an accurate prediction. '
                    f'The AI was trained on full articles, so short headlines alone are unreliable.'
                )
            }
        
        try:
            # Preprocess text
            cleaned_text = self.preprocess_text(text)
            
            if not cleaned_text:
                return {
                    'prediction': 'ERROR',
                    'confidence': 0,
                    'message': 'Text is empty after preprocessing'
                }
            
            # Vectorize text
            text_vector = self.vectorizer.transform([cleaned_text])
            
            # Get predictions from all models
            predictions = {}
            votes = {'FAKE': 0, 'REAL': 0}
            
            for model_name, model in self.models.items():
                pred = model.predict(text_vector)[0]
                pred_label = 'FAKE' if pred == 1 else 'REAL'
                predictions[model_name] = pred_label
                votes[pred_label] += 1
            
            # Majority voting
            final_prediction = 'FAKE' if votes['FAKE'] >= 2 else 'REAL'
            
            # Calculate confidence based on vote agreement
            # 3/3 agreement = 100%, 2/3 = 67%
            max_votes = max(votes['FAKE'], votes['REAL'])
            confidence = (max_votes / 3) * 100
            
            return {
                'prediction': final_prediction,
                'confidence': round(confidence, 2),
                'individual_predictions': predictions,
                'vote_count': votes,
                'ensemble_method': 'Majority Voting (3 Models)'
            }
            
        except Exception as e:
            return {
                'prediction': 'ERROR',
                'confidence': 0,
                'message': f'Prediction error: {str(e)}'
            }


# Global predictor instance
_predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = FakeNewsPredictor()
    return _predictor


def predict_news(text):
    predictor = get_predictor()
    result = predictor.predict(text)
    return result['prediction']


def predict_news_detailed(text):
    predictor = get_predictor()
    return predictor.predict(text)


# For testing
if __name__ == '__main__':
    predictor = FakeNewsPredictor()
    
    if predictor.models_loaded:
        test_text = """
        Scientists have discovered that eating chocolate every day can help you lose weight.
        A new study shows that people who ate chocolate daily lost 20 pounds in one month
        without any exercise or diet changes whatsoever according to researchers.
        """
        
        result = predictor.predict(test_text)
        print("\nPrediction Result:")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Individual Models: {result['individual_predictions']}")
        print(f"Vote Count: {result['vote_count']}")
    else:
        print("\nModels not loaded. Please train models first:")
        print("python train_isot_models.py")