from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid
from django.template.loader import render_to_string
# Create your models here.
from django.core.files.base import ContentFile
from weasyprint import HTML
from django.template.loader import render_to_string
import tempfile
import os
from django.conf import settings

class CustUser(AbstractUser):
    phone=models.IntegerField(null=True)
    is_student=models.BooleanField(default=False)
    
    
class ChatHistory(models.Model):
    user = models.ForeignKey(CustUser, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    message_type = models.CharField(max_length=10, choices=[
        ('USER', 'User Message'),
        ('BOT', 'Bot Response'),
        ('IMAGE', 'Image Upload'),
        ('REPORT', 'Medical Report')
    ])
    text_content = models.TextField(null=True, blank=True)
    image = models.ImageField(upload_to='chat_images/', null=True, blank=True)
    prediction = models.CharField(max_length=100, null=True, blank=True)
    report_file = models.FileField(upload_to='reports/', null=True, blank=True)
    report_id = models.CharField(max_length=50, unique=True, null=True, blank=True)

    class Meta:
        ordering = ['timestamp']

    