from django.urls import path
from django.urls import re_path
from . import views
from django.contrib.auth import views as auth_view
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('profile/', views.profile, name='profile'),
    path('login/', auth_view.LoginView.as_view(template_name='users/login.html'), name="login"),
    path('logout/', auth_view.LogoutView.as_view(template_name='users/logout.html'), name="logout"),
    path('discover-more/', views.discover_more, name='discover_more'),
    path('upload/', views.image_upload, name='image_upload'),
    path('process/', views.process_image, name='process_image'),
    re_path(r'^edit-parameters/(?P<filename>.+)$', views.edit_parameters, name='edit_parameters'),
    path('choose-parameters/<str:filename>/', views.choose_parameters, name='choose_parameters'),
    path('profile/', views.profile, name='profile'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
    path('sessions/', views.view_sessions, name='view_sessions'),
    path('delete-session/<int:session_id>/', views.delete_session, name='delete_session'),
    path('comparison_report/', views.comparison_report, name='comparison_report'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)