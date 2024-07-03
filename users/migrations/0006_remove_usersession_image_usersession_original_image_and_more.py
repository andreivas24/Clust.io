# Generated by Django 5.0.6 on 2024-05-29 22:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0005_usersession'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='usersession',
            name='image',
        ),
        migrations.AddField(
            model_name='usersession',
            name='original_image',
            field=models.ImageField(blank=True, null=True, upload_to='user_sessions/original_images/'),
        ),
        migrations.AlterField(
            model_name='usersession',
            name='algorithm',
            field=models.CharField(default='default_algorithm', max_length=100),
        ),
        migrations.AlterField(
            model_name='usersession',
            name='parameters',
            field=models.JSONField(default=dict),
        ),
        migrations.AlterField(
            model_name='usersession',
            name='processed_image',
            field=models.ImageField(blank=True, null=True, upload_to='user_sessions/processed_images/'),
        ),
    ]