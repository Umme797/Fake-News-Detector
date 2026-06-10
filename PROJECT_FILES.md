# Project Files Overview

## 📂 Main Files

| File | Purpose | Important? |
|------|---------|------------|
| `manage.py` | Django management script | ✅ YES |
| `requirements.txt` | Python packages needed | ✅ YES |
| `train_isot_models.py` | Train the AI models | ✅ YES |
| `README.md` | Full project documentation | ✅ YES |
| `SETUP.md` | Quick setup guide | ✅ YES |
| `.gitignore` | Git ignore rules | Optional |

## 📁 Folders

### FND_app/ - Main Application
- `models.py` - Database structure
- `views.py` - Page logic and ML integration
- `urls.py` - URL routing
- `ml_model.py` - AI prediction engine
- `ml_models/` - Trained model files (created after training)
- `admin.py` - Django admin config
- `apps.py` - App configuration

### FND_proj/ - Project Settings
- `settings.py` - Django configuration
- `urls.py` - Main URL routing
- `wsgi.py` - Server configuration
- `asgi.py` - Async server config

### templates/ - HTML Pages
- `dashboard.html` - Main interface
- `login.html` - Login page
- `register.html` - Registration page

### datasets/ - Training Data
- `True.csv` - 21,417 real news articles
- `Fake.csv` - 23,481 fake news articles
- `README.md` - Dataset information

### static/ - Static Files
- Empty folder for CSS/JS/images (optional)

## 🗑️ What Was Removed

These files were removed to keep the project clean:
- ❌ Old documentation files (PROJECT_SUMMARY.md, ML_MODELS_GUIDE.md, QUICKSTART.md)
- ❌ Test files (test_models.py)
- ❌ Old dataset (fake_news_dataset.csv with 40 articles)
- ❌ Obsolete training script (train_models.py)
- ❌ Dataset analysis scripts (analyze_dataset.py, create_sample_dataset.py)
- ❌ Original text files (FND.txt, FND.docx)
- ❌ Database file (db.sqlite3) - will be recreated on first run
- ❌ Python cache (__pycache__) - automatically recreated

## 📦 What to Share

When sharing with your friend, include:
- ✅ All folders (FND_app, FND_proj, templates, datasets)
- ✅ All main files (manage.py, requirements.txt, train_isot_models.py)
- ✅ Documentation files (README.md, SETUP.md)
- ✅ .gitignore

**Total size**: ~150 MB (mostly the datasets)

## 🚀 First Time Setup

Your friend should:
1. Extract the ZIP file
2. Follow instructions in **SETUP.md**
3. Run: `pip install -r requirements.txt`
4. Run: `python train_isot_models.py` (takes 5-10 minutes)
5. Run: `python manage.py migrate`
6. Run: `python manage.py runserver`

Done! 🎉

## 💡 Notes

- The `ml_models/` folder will be created after training
- Database file `db.sqlite3` will be created after migrations
- No need to include trained models in ZIP (they can train their own)
- All unnecessary files have been removed for clarity
