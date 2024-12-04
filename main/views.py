from django.forms import BaseModelForm
from django.shortcuts import render,redirect
from django.http import HttpResponse, JsonResponse
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from django.views.generic import TemplateView,FormView,CreateView,View
from django.urls import reverse_lazy
from django.contrib.auth import authenticate,login
from .models import *
from .forms import *
from django.contrib.auth import logout as auth_logout
from django.contrib.messages.views import SuccessMessageMixin
from django.contrib import messages
import cv2
import numpy as np
from .respiratory import predicted_class,load_and_prep_image


class_labels = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

model_path= os.path.join(os.getcwd(),'Respiratory.h5')



class LoginView(FormView):
    template_name="login.html"
    form_class=LogForm
    def post(self,request,*args,**kwargs):
        log_form=LogForm(data=request.POST)
        if log_form.is_valid():  
            us=log_form.cleaned_data.get('username')
            ps=log_form.cleaned_data.get('password')
            user=authenticate(request,username=us,password=ps)
            if user: 
                login(request,user)
                return redirect('h')
            else:
                return render(request,'login.html',{"form":log_form})
        else:
            return render(request,'login.html',{"form":log_form}) 
        

class RegView(SuccessMessageMixin,CreateView):
     form_class=RegForm
     template_name="reg.html"
     model=CustUser
     success_url=reverse_lazy("log")  
     success_message="Registration Successful!!"

     def form_invalid(self, form):
         messages.error(self.request,"Registration failed. Please correct the errors below!")
         return super().form_invalid(form)





class Home(TemplateView):
    template_name='home.html'


class PredictionView(View):
    def get(self, request):
        return render(request, "chatbot.html")
    def post(self, request):
        image = request.FILES.get('image')     
        if not image:
            return render(request, "chatbot.html", {'error': 'Please upload an image.'})    
        try:
            # Convert the uploaded image to a numpy array
            file_bytes = np.frombuffer(image.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Validate if the image is grayscale
            if len(img.shape) == 3 and img.shape[2] == 3:  # Color image has 3 channels
                b, g, r = cv2.split(img)
                if not np.array_equal(b, g) or not np.array_equal(g, r):
                    return JsonResponse({"error": "Uploaded image is not a grayscale X-ray."})
            
        except Exception as e:
            return JsonResponse({"error": f"Invalid image file: {str(e)}"}, status=400)

        try:
            test_image = load_and_prep_image(image, img_shape=128)
            cat = predicted_class(class_labels, model_path, test_image)
            user = request.user.is_student
            print(user)
            return JsonResponse({'response': cat, 'user': user})
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    



def custom_logout(request):
    auth_logout(request)
    return redirect('log')