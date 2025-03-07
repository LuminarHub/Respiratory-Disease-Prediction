# Generated by Django 5.1.3 on 2025-02-20 12:41

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0003_custuser_is_student'),
    ]

    operations = [
        migrations.CreateModel(
            name='ChatHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('message_type', models.CharField(choices=[('USER', 'User Message'), ('BOT', 'Bot Response'), ('IMAGE', 'Image Upload'), ('REPORT', 'Medical Report')], max_length=10)),
                ('text_content', models.TextField(blank=True, null=True)),
                ('image', models.ImageField(blank=True, null=True, upload_to='chat_images/')),
                ('prediction', models.CharField(blank=True, max_length=100, null=True)),
                ('report_file', models.FileField(blank=True, null=True, upload_to='medical_reports/')),
                ('report_id', models.CharField(blank=True, max_length=50, null=True, unique=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['timestamp'],
            },
        ),
    ]
