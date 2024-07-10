from django.conf import settings
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    bio = models.TextField(max_length=500, blank=True)
    image = models.ImageField(default='default-avatar.png', upload_to='profile_pics')

    def __str__(self):
        return f'{self.user.username} Profile'

# Signal to create or update the user profile
@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    else:
        instance.profile.save()

class UserSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    original_image = models.ImageField(upload_to='user_sessions/original_images/', default='path/to/default/image.jpg')
    processed_image = models.ImageField(upload_to='user_sessions/processed_images/', null=True, blank=True)
    algorithm = models.CharField(max_length=100)
    parameters = models.JSONField()
    crop_coords = models.CharField(max_length=100, null=True, blank=True)
    resize_dims = models.CharField(max_length=100, null=True, blank=True)
    filter_type = models.CharField(max_length=50, null=True, blank=True)
    processing_time = models.FloatField(null=True, blank=True)  # Add this line
    silhouette_score = models.FloatField(null=True, blank=True)  # Add this line
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Session by {self.user.username} using {self.algorithm} at {self.created_at}"

