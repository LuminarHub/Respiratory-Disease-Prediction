from django import forms
from .models import *

from django.contrib.auth.forms import UserCreationForm


class LogForm(forms.Form):
    username=forms.CharField(widget=forms.TextInput(attrs={"placeholder":"Username","class":"form-control","style":"border-radius: 0.75rem; "}))
    password=forms.CharField(widget=forms.PasswordInput(attrs={"placeholder":"Password","class":"form-control","style":"border-radius: 0.75rem; "}))


class RegForm(UserCreationForm):
    class Meta:
        model=CustUser
        fields=['email','is_student','username','password1','password2']   
    # first_name=forms.CharField(widget=forms.TextInput(attrs={"placeholder":"Firstname","class":"form-control","style":"border-radius: 0.5rem;padding:10px 50px;"}))
    # last_name=forms.CharField(widget=forms.TextInput(attrs={"placeholder":"Lastname","class":"form-control","style":"border-radius: 0.5rem;padding:10px 50px; "}))    
    username=forms.CharField(widget=forms.TextInput(attrs={"placeholder":"Username","class":"form-control","style":"border-radius: 0.5rem;padding:10px 50px; "}))
    email=forms.CharField(widget=forms.TextInput(attrs={"placeholder":"Email","class":"form-control","style":"border-radius: 0.5rem;padding:10px 50px; "}))
    is_student = forms.BooleanField(required=False)
    password1=forms.CharField(widget=forms.PasswordInput(attrs={"placeholder":"Password","class":"form-control","style":"border-radius: 0.5rem;padding:10px 50px; "}))
    password2=forms.CharField(widget=forms.PasswordInput(attrs={"placeholder":"Confirm Password","class":"form-control","style":"border-radius: 0.5rem;padding:10px 50px; "})) 


