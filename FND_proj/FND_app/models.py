from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class NewsPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    news_text = models.TextField()
    prediction = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.prediction
