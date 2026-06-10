<<<<<<< HEAD
# Fake News Detector 🔍

An AI-powered fake news detection system using Django and Machine Learning. Achieves **99.2% accuracy** using an ensemble of three ML models trained on the ISOT dataset (44,898 articles).

## ✨ Features

- **Three ML Models**: Naive Bayes, Decision Tree, and SVM working together
- **Ensemble Voting**: Combines predictions for higher accuracy
- **User Authentication**: Secure registration and login system
- **Prediction History**: Track all your news checks
- **Modern UI**: Beautiful gradient design with real-time results
- **Individual Model Votes**: See how each model voted
- **Confidence Score**: Know how certain the prediction is

## 📊 Model Performance

Trained on ISOT Dataset (44,898 news articles):
- **Naive Bayes**: 94.80% accuracy
- **Decision Tree**: 99.70% accuracy
- **SVM**: 98.80% accuracy
- **Ensemble**: 99.20% accuracy

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models

```bash
python train_isot_models.py
```

This will train all three ML models using the ISOT dataset and save them in `FND_app/ml_models/`. Training takes about 5-10 minutes.

### 3. Set Up Database

```bash
python manage.py migrate
```

### 4. Run the Server

```bash
python manage.py runserver
```

Visit: http://127.0.0.1:8000/

## 📁 Project Structure

```
FND/
├── FND_app/              # Main Django application
│   ├── models.py         # Database models
│   ├── views.py          # View functions
│   ├── ml_model.py       # ML prediction engine
│   └── ml_models/        # Trained model files (.pkl)
├── FND_proj/             # Django project settings
│   └── settings.py       # Configuration
├── templates/            # HTML templates
│   ├── dashboard.html    # Main interface
│   ├── login.html        # Login page
│   └── register.html     # Registration page
├── datasets/             # Training data
│   ├── True.csv          # 21,417 real news articles
│   └── Fake.csv          # 23,481 fake news articles
├── train_isot_models.py  # Model training script
├── requirements.txt      # Python dependencies
└── manage.py             # Django management script
```

## 🎯 How to Use

1. **Register**: Create a new account at `/register/`
2. **Login**: Sign in with your credentials
3. **Enter News**: Paste any news article text
4. **Get Results**: See if it's REAL or FAKE with confidence score
5. **View Details**: Check individual model votes
6. **Check History**: See all your previous predictions

## 🔧 Technologies Used

- **Backend**: Django 6.0.2
- **Machine Learning**: scikit-learn, pandas, numpy
- **Database**: SQLite (development) / MySQL (production)
- **Frontend**: HTML5, CSS3, JavaScript

## 📝 About the Dataset

**ISOT Fake News Dataset**:
- 44,898 total articles
- 21,417 real news articles
- 23,481 fake news articles
- Collected from Reuters (real) and various sources (fake)

## 🎨 UI Features

- Gradient backgrounds with animations
- Color-coded results (Green = REAL, Red = FAKE)
- Progress bars for confidence visualization
- Responsive design for all devices
- Clean, modern interface

## 🔐 Security Features

- User authentication required
- CSRF protection enabled
- Secure password hashing
- Session management

## 📌 Notes

- First-time setup requires model training (5-10 minutes)
- Models are saved and reused for fast predictions
- SQLite is used by default for easy setup
- For production, switch to MySQL in settings.py

## 🤝 Contributing

Feel free to fork this project and make improvements!

## 📄 License

This project is open source and available for educational purposes.

---

**Developed with ❤️ using Django and Machine Learning

result = predict_news_detailed("Breaking: Scientists discover cure for everything!")
print(result)
# Output: {'prediction': 'FAKE', 'confidence': 100.0, ...}
```

## Troubleshooting

### Models Not Loaded
**Error**: "Models not trained yet"  
**Fix**: Run `python train_models.py`

### Low Prediction Accuracy
**Fix**: Get larger dataset and retrain

### MySQL Connection Error
**For testing only**: Change `settings.py` to use SQLite:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

## Notes

- The ML prediction feature is currently commented out in `views.py`
- Make sure to secure your `SECRET_KEY` in production
- Update database credentials before deployment
- The `ml_model.py` file needs to be created for news prediction functionality
=======
## 📰 Fake News Detector

A Django-based web application that uses Natural Language Processing (NLP) 
and Machine Learning to classify news articles as Real or Fake.

### 🔍 About the Project
With the rise of misinformation, detecting fake news has become more important 
than ever. This project leverages trained ML models to analyze news content 
and predict its authenticity in real time.

### ⚙️ Models Used
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- TF-IDF Vectorizer for text preprocessing

### 🛠️ Tech Stack
- **Backend:** Python, Django
- **Machine Learning:** Scikit-learn, NLTK
- **Frontend:** HTML, CSS, Bootstrap
- **Authentication:** Django Allauth (Email + Google Login)
- **Database:** SQLite
- **Dataset:** ISOT Fake News Dataset
>>>>>>> e9e1e02bcba6f89e2d8ee4c1c190a81305b089e3
