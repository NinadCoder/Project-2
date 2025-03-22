from django.contrib import admin
from django.urls import path
from home import views
urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    #path('contact/', views.contact, name='contact'),
    path('prediction/', views.prediction, name='prediction'),
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register')
]
