# SETUP INSTRUCTIONS

## 🚀 Quick Setup Guide

Follow these steps to get the Fake News Detector running on your machine:

### Step 1: Install Python Packages

```bash
pip install -r requirements.txt
```

This will install:
- Django 6.0.2
- scikit-learn
- pandas
- numpy
- joblib
- mysqlclient

### Step 2: Train the AI Models (IMPORTANT!)

**You must train the models before running the project.**

```bash
python train_isot_models.py
```

⏱️ This takes about **5-10 minutes** and will:
- Load the ISOT dataset (44,898 articles)
- Train 3 ML models (Naive Bayes, Decision Tree, SVM)
- Save models to `FND_app/ml_models/` folder
- Show accuracy results (99.2% ensemble accuracy)

### Step 3: Setup Database

```bash
python manage.py migrate
```

This creates the SQLite database with all required tables.

### Step 4: Run the Server

```bash
python manage.py runserver
```

🎉 **Done!** Open your browser and go to: http://127.0.0.1:8000/

---

## 📝 First Time Using the App

1. Click **Register** to create an account
2. **Login** with your credentials
3. Paste a news article in the text box
4. Click **Check News**
5. See the AI prediction (REAL or FAKE) with confidence score!

---

## ⚠️ Common Issues

### "Model files not found"
- **Solution**: You forgot to train the models! Run: `python train_isot_models.py`

### "Module not found" errors
- **Solution**: Install dependencies: `pip install -r requirements.txt`

### Server won't start
- **Solution**: Make sure you ran migrations: `python manage.py migrate`

---

## 🔄 Retraining Models

Want to retrain with different settings? Just run:

```bash
python train_isot_models.py
```

The script will automatically overwrite old models with new ones.

---

## 💡 Tips for Your Friend

- The project uses **SQLite** by default (no database setup needed!)
- All models are saved locally after training
- The UI has a modern gradient design
- You can see individual model votes for each prediction
- Prediction history is saved for each user

---

**Need Help?** Check the main README.md for detailed information!
