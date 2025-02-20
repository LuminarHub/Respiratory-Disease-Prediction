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

import os
import uuid
import numpy as np
import cv2
from django.utils import timezone
from django.template.loader import render_to_string
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views import View
from django.views.generic import TemplateView
from weasyprint import HTML
from .models import ChatHistory  # Ensure this model exists

def generate_report_id():
    return f"RPT-{uuid.uuid4().hex[:8].upper()}"
def get_doctor_notes(prediction):
    notes = {
        'Bacterial Pneumonia': "The X-ray shows signs of inflammation in the lung tissue, consistent with pneumonia. Areas of increased opacity suggest fluid or inflammatory buildup in the air spaces.",
        'Normal': "The chest X-ray appears normal with clear lung fields. No significant abnormalities detected in the pulmonary tissue.",
        'Corona Virus Disease': "The X-ray reveals bilateral peripheral ground-glass opacities characteristic of COVID-19 pneumonia. Pattern is consistent with viral pneumonia.",
        'Tuberculosis': "The X-ray shows patchy areas of consolidation and possible cavitation, typical of pulmonary tuberculosis. Upper lobe involvement is noted.",
        'Viral Pneumonia': "The X-ray indicates diffuse interstitial infiltrates, commonly seen in viral pneumonia. No bacterial consolidation is noted."
    }
    return notes.get(prediction, "Findings require further clinical correlation.")

def get_recommendations(prediction):
    recs = {
        'Bacterial Pneumonia': [
            "Prescribed course of appropriate antibiotics if bacterial infection is confirmed",
            "Rest and adequate hydration",
            "Follow-up chest X-ray in 4-6 weeks to ensure resolution",
            "Monitor temperature and breathing difficulty",
            "Return if symptoms worsen or new symptoms develop"
        ],
        'Normal': [
            "No specific treatment required for lung condition",
            "Continue regular health maintenance",
            "Annual health check-ups recommended",
            "Practice good respiratory hygiene",
            "Maintain a healthy lifestyle"
        ],
        'Corona Virus Disease': [
            "Immediate isolation to prevent transmission",
            "Monitor oxygen saturation levels regularly",
            "Supportive care and symptom management",
            "Follow current COVID-19 treatment protocols",
            "Regular virtual check-ins with healthcare provider"
        ],
        'Tuberculosis': [
            "Begin standard TB treatment regimen",
            "Regular monitoring of liver function",
            "Contact tracing and family screening",
            "Directly Observed Therapy (DOT) implementation",
            "Monthly follow-up with chest X-rays"
        ],
        'Viral Pneumonia': [
            "Rest and maintain hydration",
            "Monitor for breathing difficulty and seek medical help if it worsens",
            "Over-the-counter fever reducers and cough suppressants as needed",
            "Isolation if contagious, to prevent virus spread",
            "Follow-up with a doctor if symptoms persist beyond two weeks"
        ]
    }
    return recs.get(prediction, ["Please consult with a healthcare provider for specific recommendations."])


def generate_medical_report(user, prediction):
    report_id = generate_report_id()
    context = {
        'report_id': report_id,
        'patient_name': user.username if user.is_authenticated else "Unknown",
        'date': timezone.now().strftime("%Y-%m-%d %H:%M:%S"),
        'prediction': prediction,
        'doctor_notes': get_doctor_notes(prediction),
        'recommendations': get_recommendations(prediction)
    }

    # Render HTML report
    report_html = render_to_string('report.html', context)

    # Generate PDF
    pdf_file = HTML(string=report_html).write_pdf()

    # Define filename and save path
    pdf_filename = f"{report_id}.pdf"
    pdf_dir = os.path.join(settings.MEDIA_ROOT, "reports")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, pdf_filename)

    # Save the PDF file
    with open(pdf_path, "wb") as f:
        f.write(pdf_file)

    # Save reference in database
    # chat_entry = ChatHistory.objects.create(
    #     user=user,
    #     message_type='REPORT',
    #     text_content=f"Generated report {report_id}",
    #     prediction=prediction,
    #     report_id=report_id,
    #     report_file=pdf_filename
    # )

    return pdf_path, pdf_filename

class PredictionView(View):
    def get(self, request):
        chat_history = ChatHistory.objects.filter(user=request.user)
        return render(request, "chatbot.html", {'chat_history': chat_history})

    def post(self, request):
        image = request.FILES.get('image')
        text = request.POST.get('userInput')

        if text:
            return self.handle_text_query(request, text)
        elif image:
            return self.handle_image_query(request, image)
        else:
            return JsonResponse({'error': 'Invalid request. Provide text or an image.'})

    def handle_text_query(self, request, text):
        try:
            ChatHistory.objects.create(user=request.user, message_type='USER', text_content=text)
            data = get_groq_response(text)

            ChatHistory.objects.create(user=request.user, message_type='BOT', text_content=data)
            return JsonResponse({'response': data})
        except Exception as e:
            return JsonResponse({'error': f'Failed to get response: {str(e)}'})

    def handle_image_query(self, request, image):
        try:
            file_bytes = np.frombuffer(image.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if len(img.shape) == 3 and img.shape[2] == 3:
                b, g, r = cv2.split(img)
                if not np.array_equal(b, g) or not np.array_equal(g, r):
                    return JsonResponse({"error": "Uploaded image is not a grayscale X-ray."})

            chat_entry = ChatHistory.objects.create(user=request.user, message_type='IMAGE', image=image)

            # Placeholder functions
            test_image = load_and_prep_image(image, img_shape=128)
            prediction = predicted_class(class_labels, model_path, test_image)

            chat_entry.prediction = prediction
            chat_entry.save()

            # Generate medical report
            pdf_path, pdf_filename = generate_medical_report(request.user, prediction)
            chat_entry.report_file = pdf_filename
            chat_entry.report_id = generate_report_id()
            chat_entry.save()

            return JsonResponse({'response': prediction, 'report_id': chat_entry.report_id})
        except Exception as e:
            return JsonResponse({'error': str(e)})

import os
from django.conf import settings
from django.http import FileResponse, HttpResponseNotFound

def download_pdf(request, filename):
    pdf_path = os.path.join(settings.MEDIA_ROOT, "reports", filename)

    if os.path.exists(pdf_path):
        return FileResponse(open(pdf_path, "rb"), content_type="application/pdf", as_attachment=True)
    else:
        return HttpResponseNotFound("File not found")

# class DownloadReportView(View):
#     def get(self, request, report_id):
#         try:
#             report = ChatHistory.objects.get(report_id=report_id, user=request.user)
#             pdf_path = os.path.join(settings.MEDIA_ROOT, "reports", report.report_file)

#             if not os.path.exists(pdf_path):
#                 return HttpResponse("Report not found", status=404)

#             with open(pdf_path, "rb") as f:
#                 response = HttpResponse(f.read(), content_type="application/pdf")
#                 response["Content-Disposition"] = f'attachment; filename="{report_id}.pdf"'
#                 return response
#         except ChatHistory.DoesNotExist:
#             return HttpResponse("Report not found", status=404)

import requests
from groq import Groq


def get_groq_response(user_input):
    """
    Communicate with the GROQ chatbot to get a response based on user input.
    """
    print("user input:", user_input)
    
    client = Groq(
        api_key="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",
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_input,  # Use the user input here
            }
        ],
        model="llama3-8b-8192",
        stream=False,
    )

    response = chat_completion.choices[0].message.content
    return response

def custom_logout(request):
    auth_logout(request)
    return redirect('log')


class DoctorsView(TemplateView):
    template_name = 'doctors.html'
    
    
import numpy as np
import cv2
from django.core.files.base import ContentFile
import base64

# class PredictionView(View):
#     def get(self, request):
#         # Get chat history for the current user
#         chat_history = ChatHistory.objects.filter(user=request.user)
#         return render(request, "chatbot.html", {'chat_history': chat_history})

#     def post(self, request):
#         image = request.FILES.get('image')     
#         text = request.POST.get('userInput')
        
#         if text:
#             try:
#                 # Save user message
#                 ChatHistory.objects.create(
#                     user=request.user,
#                     message_type='USER',
#                     text_content=text
#                 )
                
#                 # Get bot response
#                 data = get_groq_response(text)
                
#                 # Save bot response
#                 ChatHistory.objects.create(
#                     user=request.user,
#                     message_type='BOT',
#                     text_content=data
#                 )
                
#                 return JsonResponse({'groq': data})
#             except Exception as e:
#                 return JsonResponse({"error": f"Failed to get GROQ response: {str(e)}"})

#         if not image:
#             return render(request, "chatbot.html", {'error': 'Please upload an image.'})    
        
#         try:
#             file_bytes = np.frombuffer(image.read(), np.uint8)
#             img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
#             if len(img.shape) == 3 and img.shape[2] == 3: 
#                 b, g, r = cv2.split(img)
#                 if not np.array_equal(b, g) or not np.array_equal(g, r):
#                     return JsonResponse({"error": "Uploaded image is not a grayscale X-ray."})
                    
#             # Save image upload to history
#             chat_entry = ChatHistory.objects.create(
#                 user=request.user,
#                 message_type='IMAGE',
#                 image=image
#             )
            
#             test_image = load_and_prep_image(image, img_shape=128)
#             cat = predicted_class(class_labels, model_path, test_image)
            
#             # Update the chat entry with prediction
#             chat_entry.prediction = cat
#             chat_entry.save()
            
#             user = request.user.is_student
#             return JsonResponse({'response': cat, 'user': user})
        
#         except Exception as e:
#             return JsonResponse({'error': str(e)})

class HistoryView(View):
    def get(self, request):
        chat_history = ChatHistory.objects.filter(user=request.user).order_by('timestamp')
        return render(request, "history.html", {'chat_history': chat_history})
    
    
class HistoryAllView(View):
    def get(self, request):
        chat_history = ChatHistory.objects.all().order_by('timestamp')
        return render(request, "history_all.html", {'chat_history': chat_history})

def delete_chat(request,chat_id):
    """View for deleting files"""
    try:
        file = ChatHistory.objects.get(id=chat_id)
        file.delete()
        messages.success(request, 'ChatHistory deleted successfully!')
    except ChatHistory.DoesNotExist:
        messages.error(request, 'ChatHistory not found!')
    except Exception as e:
        messages.error(request, f'Error deleting file: {str(e)}')
        
    return redirect('chat_history')