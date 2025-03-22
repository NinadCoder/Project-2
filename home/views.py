from django.shortcuts import render
from keras_preprocessing import image
import numpy as np
from keras import models
from django.core.files.storage import FileSystemStorage

model = models.load_model(r"C:\Users\ninad\OneDrive\Desktop\Deep Learning Project\dogcat\static\cat_or_dog.h5")

# Create your views here.
def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def login(request):
    return render(request, 'login.html')

def register(request):
    return render(request, 'register.html')

#def contact(request):
    #return render(request, 'contact.html')

def prediction(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Save the uploaded file
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(filename)
        
        # Predict label (Dog/Cat)
        label = predict_image(fs.path(filename))  # returns "Dog" or "Cat"
        
        return render(request, 'result.html', {
            'prediction': label,
            'file_url': file_url
        })
    
    # If GET request, just render the upload page
    return render(request, 'prediction.html')

from keras_preprocessing.image import load_img, img_to_array

def predict_image(img_path):
    img = load_img(img_path, target_size=(64, 64))  # Adjust as per your model
    input_arr = img_to_array(img)
    input_arr = np.expand_dims(input_arr, axis=0)
    input_arr /= 255.0  # normalize if your model was trained with normalization
    
    prediction = model.predict(input_arr)  # e.g. [[0.92017883]]
    prob = prediction[0][0]
    
    if prob >= 0.5:
        return "Dog"
    else:
        return "Cat"