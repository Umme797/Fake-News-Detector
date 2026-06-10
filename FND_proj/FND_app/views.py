from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required

from .models import NewsPrediction
from .ml_model import predict_news_detailed


# REGISTER
def register_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        if password1 != password2:
            messages.error(request, "Passwords do not match")
            return redirect("register")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return redirect("register")

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1
        )
        user.save()
        messages.success(request, "Registration successful")
        return redirect("login")

    return render(request, "register.html")


# LOGIN
def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid credentials")
            return redirect("login")

    return render(request, "login.html")


# LOGOUT
def logout_view(request):
    logout(request)
    return redirect("login")


# DASHBOARD
@login_required
def dashboard(request):
    result = None
    confidence = None
    individual_predictions = None
    vote_count = None
    prediction_message = None

    if request.method == "POST":
        news_text = request.POST.get("news_text", "").strip()

        if news_text:
            prediction_result = predict_news_detailed(news_text)
            result = prediction_result['prediction']

            if result == 'TOO_SHORT':
                # Don't save to DB — not a real prediction
                prediction_message = prediction_result.get('message', '')

            elif result == 'ERROR':
                prediction_message = prediction_result.get('message', 'An error occurred.')

            else:
                confidence = prediction_result.get('confidence', 0)
                individual_predictions = prediction_result.get('individual_predictions', {})
                vote_count = prediction_result.get('vote_count', {})

                # Save to database
                NewsPrediction.objects.create(
                    user=request.user,
                    news_text=news_text,
                    prediction=result
                )

    history = NewsPrediction.objects.filter(user=request.user).order_by('-created_at')

    return render(request, "dashboard.html", {
        "result": result,
        "confidence": confidence,
        "individual_predictions": individual_predictions,
        "vote_count": vote_count,
        "prediction_message": prediction_message,
        "history": history,
    })