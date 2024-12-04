from django.db import models
from django.contrib.auth.models import AbstractUser
# Create your models here.


class CustUser(AbstractUser):
    phone=models.IntegerField(null=True)
    is_student=models.BooleanField(default=False)