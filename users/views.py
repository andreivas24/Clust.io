from io import BytesIO
from django.conf import settings
from django.shortcuts import render, redirect
from users.models import UserSession
from .forms import UserRegisterForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import ProfileForm
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
import os
from PIL import Image, ImageEnhance, ImageFilter
from .algorithms import downsample_image, process_and_smooth, run_agglomerative_clustering, run_birch, run_kmeans, run_gmm, run_mini_batch_kmeans, save_plot_2d, save_plot_3d
import numpy as np
from django.core.files.base import ContentFile
from django.views.decorators.cache import never_cache
import logging
import time
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from django.core.files.base import ContentFile
from sklearn.metrics import silhouette_score

def home(request):
    return render(request, 'users/home.html')

def register(request):
    if request.method == "POST":
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Hi {username}, your account was created successfully')
            return redirect('home')
    else:
        form = UserRegisterForm()

    return render(request, 'users/register.html', {'form': form})


@login_required()
def profile(request):
    return render(request, 'users/profile.html')

def logged_out(request):
    return render(request, 'logout.html')

def discover_more(request):
    return render(request, 'users/discover_more.html')

#For image upload
@never_cache
@login_required()
def image_upload(request):
    if request.method == 'POST':
        logging.debug("POST data: %s", request.POST)
        logging.debug("FILES data: %s", request.FILES)
        if 'image_file' in request.FILES:
            file_obj = request.FILES['image_file']
            
            # Open the uploaded image file
            image = Image.open(file_obj)
            
            # Resize the image if necessary
            if image.size[0] > 400 or image.size[1] > 400:
                image = resize_image(image, max_size=(800, 800))
            
            # Determine the format to save the image
            format = image.format if image.format else 'JPEG'
            extension = file_obj.name.split('.')[-1]
            if extension.lower() not in ['jpg', 'jpeg', 'png']:
                format = 'JPEG'
            
            # Save the resized image to a BytesIO object
            img_io = BytesIO()
            image.save(img_io, format=format)
            img_io.seek(0)
            
            # Create a new InMemoryUploadedFile from the BytesIO object
            file_obj = InMemoryUploadedFile(
                img_io, None, file_obj.name, file_obj.content_type, img_io.tell, None
            )
            
            # Save the resized image file
            fs = FileSystemStorage()
            filename = fs.save(file_obj.name, file_obj)
            file_url = fs.url(filename)
            
            # Pass the URL to the choose_parameters view
            return redirect('choose_parameters', filename=filename)
        else:
            return render(request, 'users/upload.html', {'error': 'No file was selected.'})
    return render(request, 'users/upload.html')

@login_required()
def choose_parameters(request, filename):
    fs = FileSystemStorage()
    file_url = fs.url(filename)
    request.session['original_image_path'] = filename
    return render(request, 'users/choose_parameters.html', {
        'file_url': file_url,
        'filename': filename,
    })

logger = logging.getLogger(__name__)

@login_required()
def edit_parameters(request, filename):
    fs = FileSystemStorage()
    original_image_path = request.session.get('original_image_path', None)
    original_image_url = fs.url(original_image_path) if original_image_path else None

    return render(request, 'users/edit_parameters.html', {
        'original_image_url': original_image_url,
        'original_filename': original_image_path,
        'filename': filename,
    })

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Usage
file_path = 'D:\\Facultate\\LicentaV2\\proiectLicentaV3\\proiectLicentaV3\\licenceWebsite\\media\\processed_images'
ensure_dir(file_path)

def resize_image(image, max_size=(800, 800)):
    original_size = image.size
    ratio = min(max_size[0] / original_size[0], max_size[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    resized_image = image.resize(new_size, Image.LANCZOS)
    
    # Sharpen the image slightly after resizing to improve quality
    enhancer = ImageEnhance.Sharpness(resized_image)
    enhanced_image = enhancer.enhance(1.9)  # Adjust the factor as needed

    return enhanced_image

def custom_resize_image(image, new_size):
    resized_image = image.resize(new_size, Image.LANCZOS)
    
    # Sharpen the image slightly after resizing to improve quality
    enhancer = ImageEnhance.Sharpness(resized_image)
    enhanced_image = enhancer.enhance(1.9)  # Adjust the factor as needed

    return enhanced_image

def process_image(request):
    if request.method == 'POST':
        k = int(request.POST.get('k', '4'))
        sigma = float(request.POST.get('sigma')) if request.POST.get('sigma') else None
        downsample_factor = float(request.POST.get('downsample_factor')) if request.POST.get('downsample_factor') else None
        algorithm = request.POST.get('algorithm', 'kmeans')
        original_filename = request.POST.get('original_filename', '')
        processed_filename = request.POST.get('filename', '')

        crop_coords = request.POST.get('crop', '')
        resize_dims = request.POST.get('resize', '')
        filter_type = request.POST.get('filter', 'None')

        if not original_filename:
            return HttpResponse("No image file specified", status=400)

        fs = FileSystemStorage()
        image_path = fs.path(original_filename)

        if not os.path.exists(image_path):
            return HttpResponse("File not found.", status=404)

        try:
            start_time = time.time()
            
            image = Image.open(image_path)
            original_size = image.size  # Store original size

            # Apply cropping if specified
            if crop_coords:
                try:
                    left, upper, right, lower = map(int, crop_coords.split(','))
                    # Ensure valid cropping coordinates
                    if left < 0 or upper < 0 or right > original_size[0] or lower > original_size[1] or left >= right or upper >= lower:
                        return HttpResponse(f"Invalid crop coordinates: {crop_coords}. Please ensure they are within the image dimensions.", status=400)
                    # Ensure the cropped area is not too small
                    if (right - left) < 50 or (lower - upper) < 50:
                        return HttpResponse(f"Cropped area too small: {crop_coords}. Please select a larger area.", status=400)
                    image = image.crop((left, upper, right, lower))
                except Exception as e:
                    return HttpResponse(f"Invalid crop coordinates: {crop_coords}. Error: {str(e)}", status=400)

            # Apply custom resizing if specified
            if resize_dims:
                try:
                    width, height = map(int, resize_dims.split(','))
                    image = custom_resize_image(image, (width, height))
                    original_size = (width, height)  # Update the original size to the resized dimensions
                except Exception as e:
                    return HttpResponse(f"Invalid resize dimensions: {resize_dims}. Error: {str(e)}", status=400)
            else:
                # Apply default resizing if the image is too large
                image = resize_image(image, max_size=(800, 800))

            # Apply filter if specified
            if filter_type != 'None':
                try:
                    filter_map = {
                        'BLUR': ImageFilter.BLUR,
                        'CONTOUR': ImageFilter.CONTOUR,
                        'DETAIL': ImageFilter.DETAIL,
                        'EDGE_ENHANCE': ImageFilter.EDGE_ENHANCE,
                        'EDGE_ENHANCE_MORE': ImageFilter.EDGE_ENHANCE_MORE,
                        'EMBOSS': ImageFilter.EMBOSS,
                        'FIND_EDGES': ImageFilter.FIND_EDGES,
                        'SHARPEN': ImageFilter.SHARPEN,
                        'SMOOTH': ImageFilter.SMOOTH,
                        'SMOOTH_MORE': ImageFilter.SMOOTH_MORE,
                    }
                    image = image.filter(filter_map[filter_type])
                except Exception as e:
                    return HttpResponse(f"Invalid filter type: {filter_type}. Error: {str(e)}", status=400)

            if image.mode == 'RGBA':
                image = image.convert('RGB')

            img_data = np.array(image)
            print(f"Original image shape: {img_data.shape}")

            # Downsample image if necessary
            if algorithm == 'agglomerative' and downsample_factor and downsample_factor < 1.0:
                img_data = downsample_image(img_data, downsample_factor)
                print(f"Downsampled image shape: {img_data.shape}")

            if algorithm == 'kmeans':
                clustered_img, labels, centers = run_kmeans(img_data, k)
            elif algorithm == 'gmm':
                clustered_img, labels, centers = run_gmm(img_data, k, sigma)
            elif algorithm == 'minibatch_kmeans':
                clustered_img, labels, centers = run_mini_batch_kmeans(img_data, k)
            elif algorithm == 'birch':
                clustered_img, labels, centers = run_birch(img_data, k)
            elif algorithm == 'agglomerative':
                clustered_img, labels, centers = run_agglomerative_clustering(img_data, k)

            print(f"Clustered image shape: {clustered_img.shape}")

            enhanced_img = process_and_smooth(clustered_img, labels, centers)
            print(f"Enhanced image shape: {enhanced_img.shape}")

            plot2d_url = save_plot_2d(img_data.reshape(-1, 3), labels, centers)
            plot3d_url = save_plot_3d(img_data.reshape(-1, 3), labels, centers)

            enhanced_img_pil = Image.fromarray(enhanced_img)
            enhanced_img_pil = enhanced_img_pil.resize(original_size, Image.Resampling.LANCZOS)

            # Convert to RGB if the image is RGBA
            if enhanced_img_pil.mode == 'RGBA':
                enhanced_img_pil = enhanced_img_pil.convert('RGB')

            buffer = BytesIO()
            enhanced_img_pil.save(buffer, format='JPEG')
            timestamp = int(time.time() * 1000)
            new_processed_filename = f'processed_{timestamp}_{processed_filename}'
            fs.save(new_processed_filename, ContentFile(buffer.getvalue()))
            processed_image_url = fs.url(new_processed_filename)

            parameters = {'k': k}
            if sigma is not None and algorithm == 'gmm':
                parameters['sigma'] = sigma
            if downsample_factor is not None and algorithm == 'agglomerative':
                parameters['downsample_factor'] = downsample_factor

            # Calculate silhouette score
            sample_pixels = img_data.reshape(-1, 3)
            sample_labels = labels.reshape(-1)
            sample_size = min(10000, len(sample_pixels))
            sample_indices = np.random.choice(len(sample_pixels), sample_size, replace=False)
            sample_pixels = sample_pixels[sample_indices]
            sample_labels = sample_labels[sample_indices]
            silhouette_avg = 2 * silhouette_score(sample_pixels, sample_labels)
            print(f"Silhouette Score: {silhouette_avg}")

            # Calculate processing time
            processing_time = time.time() - start_time
            print(f"Processing time: {processing_time}")

            save_session(request, original_image=original_filename, processed_image=new_processed_filename, algorithm=algorithm, parameters=parameters, processing_time=processing_time, silhouette_score=silhouette_avg)

            # Generate PDF report
            plot2d_paths = {
                'rg': os.path.join(settings.MEDIA_ROOT, f'plot_2d_rg.png'),
                'gb': os.path.join(settings.MEDIA_ROOT, f'plot_2d_gb.png'),
                'br': os.path.join(settings.MEDIA_ROOT, f'plot_2d_br.png')
            }
            plot3d_path = os.path.join(settings.MEDIA_ROOT, 'plot_3d.html')
            report_filename = f'report_{timestamp}.pdf'
            report_filepath = os.path.join(settings.MEDIA_ROOT, report_filename)

            generate_pdf_report(
                algorithm=algorithm,
                k=k,
                processed_image_path=os.path.join(settings.MEDIA_ROOT, new_processed_filename),
                plot_2d_paths=plot2d_paths,
                plot_3d_path=plot3d_path,
                output_path=report_filepath,
                sigma=sigma if algorithm == 'gmm' else None,
                downsample_factor=downsample_factor if algorithm == 'agglomerative' else None,
                silhouette_score=silhouette_avg,
                processing_time=processing_time
            )

            pdf_url = fs.url(report_filename)

            return render(request, 'users/processed_image.html', {
                'processed_image_url': processed_image_url,
                'plot3d_url': plot3d_url,
                'plot2d_url_rg': plot2d_url['rg'],
                'plot2d_url_gb': plot2d_url['gb'],
                'plot2d_url_br': plot2d_url['br'],
                'algorithm': algorithm,
                'k': k,
                'sigma': sigma if algorithm == 'gmm' else None,
                'downsample_factor': downsample_factor if algorithm == 'agglomerative' else None,
                'filename': new_processed_filename,
                'original_filename': original_filename,
                'pdf_url': pdf_url,
                'processing_time': processing_time,
                'silhouette_score': silhouette_avg
            })
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return HttpResponse(f"Error processing image: {str(e)}", status=500)

def recolor_image(image_data, labels, centers):
    # Create an image where each pixel is replaced by its corresponding cluster center
    new_image_data = centers[labels].reshape(image_data.shape).astype(np.uint8)
    return new_image_data

def save_image(processed_image_array, original_filename):
    fs = FileSystemStorage()
    # Ensure the array is of type uint8
    image = Image.fromarray(np.uint8(processed_image_array))

    # Prepare the buffer to save the image
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)  # Move to the start of the buffer

    # Generate a new filename based on the original
    new_filename = 'processed_' + original_filename

    # Save the image to the FileSystemStorage
    file_path = fs.save(new_filename, ContentFile(buffer.getvalue()))

    # Generate the URL to access the file
    return fs.url(file_path)

def image_parameters(request, filename):
    if request.method == 'POST':
        # Process the form with new parameters
        k = int(request.POST.get('k', 4))
        algorithm = request.POST.get('algorithm', 'kmeans')
        fs = FileSystemStorage()
        image_path = fs.path(filename)
        
        if not os.path.exists(image_path):
            return HttpResponse("File not found", status=404)
        
        try:
            image = Image.open(image_path)
            img_data = np.array(image)
            if algorithm == 'kmeans':
                processed_img_data = run_kmeans(img_data, k)
            elif algorithm == 'gmm':
                sigma = float(request.POST.get('sigma', 10))
                processed_img_data = run_gmm(img_data, k, sigma)
            processed_image = Image.fromarray(processed_img_data)
            buffer = BytesIO()
            processed_image.save(buffer, format='JPEG')
            buffer.seek(0)
            fs.save(filename, ContentFile(buffer.read()))
            processed_image_url = fs.url(filename)
            
            return render(request, 'users/processed_image.html', {
                'processed_image_url': processed_image_url,
                'algorithm': algorithm,
                'k': k,
                'sigma': sigma if algorithm == 'gmm' else None
            })
        except Exception as e:
            return HttpResponse(f"Error processing image: {str(e)}", status=500)
    else:
        # Display form with existing parameters
        return render(request, 'users/change_parameters.html', {
            'filename': filename
        })

#For profile
def profile(request):
    print(request.user.profile.image.url)  # This should print the URL path to the console
    return render(request, 'users/profile.html', {'user': request.user})

@login_required
def edit_profile(request):
    if request.method == 'POST':
        form = ProfileForm(request.POST, request.FILES, instance=request.user.profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your profile has been updated!')
            return redirect('profile')
    else:
        form = ProfileForm(instance=request.user.profile)
    return render(request, 'users/edit_profile.html', {'form': form})

@login_required
def save_session(request, original_image, processed_image, algorithm, parameters, processing_time, silhouette_score):
    user = request.user
    session = UserSession(
        user=user,
        original_image=original_image,
        processed_image=processed_image,
        algorithm=algorithm,
        parameters=parameters,
        crop_coords=request.POST.get('crop', ''),
        resize_dims=request.POST.get('resize', ''),
        filter_type=request.POST.get('filter', 'None'),
        processing_time=processing_time,  # Add this line
        silhouette_score=silhouette_score  # Add this line
    )
    session.save()

@login_required
def view_sessions(request):
    sessions = UserSession.objects.filter(user=request.user)
    return render(request, 'users/sessions.html', {'sessions': sessions})

def delete_session(request, session_id):
    session = get_object_or_404(UserSession, id=session_id, user=request.user)
    session.delete()
    return redirect(reverse('view_sessions'))

def generate_pdf_report(algorithm, k, processed_image_path, plot_2d_paths, plot_3d_path, output_path, sigma=None, downsample_factor=None, silhouette_score=None, processing_time=None):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []

    # Add the title
    Story.append(Paragraph("Image Processing Report", styles['Title']))
    Story.append(Spacer(1, 12))

    # Add the parameters
    Story.append(Paragraph(f"Algorithm: {algorithm}", styles['Normal']))
    Story.append(Paragraph(f"k: {k}", styles['Normal']))
    if sigma is not None:
        Story.append(Paragraph(f"Sigma: {sigma}", styles['Normal']))
    if downsample_factor is not None:
        Story.append(Paragraph(f"Downsample Factor: {downsample_factor}", styles['Normal']))
    if processing_time is not None:
        Story.append(Paragraph(f"Processing Time: {processing_time:.2f} seconds", styles['Normal']))
    if silhouette_score is not None:
        Story.append(Paragraph(f"Silhouette Score: {silhouette_score:.4f}", styles['Normal']))
    Story.append(Spacer(1, 12))

    # Add the processed image
    Story.append(Paragraph("Processed Image:", styles['Normal']))
    img = Image(processed_image_path)
    img.drawHeight = 200  # Adjust the height
    img.drawWidth = 200  # Adjust the width
    Story.append(img)
    Story.append(Spacer(1, 12))

    # Add the 2D plots
    Story.append(Paragraph("2D Plot RG:", styles['Normal']))
    img_rg = Image(plot_2d_paths['rg'])
    img_rg.drawHeight = 300  # Adjust the height
    img_rg.drawWidth = 300  # Adjust the width
    Story.append(img_rg)

    Story.append(Spacer(1, 12))
    Story.append(Paragraph("2D Plot GB:", styles['Normal']))
    img_gb = Image(plot_2d_paths['gb'])
    img_gb.drawHeight = 300  # Adjust the height
    img_gb.drawWidth = 300  # Adjust the width
    Story.append(img_gb)

    Story.append(Spacer(1, 12))
    Story.append(Paragraph("2D Plot BR:", styles['Normal']))
    img_br = Image(plot_2d_paths['br'])
    img_br.drawHeight = 300  # Adjust the height
    img_br.drawWidth = 300  # Adjust the width
    Story.append(img_br)

    Story.append(Spacer(1, 12))

    # Add the 3D plot link
    Story.append(Paragraph("3D Plot:", styles['Normal']))
    try:
        # Add a hyperlink to the HTML file instead of embedding it
        Story.append(Paragraph(f"<a href='{plot_3d_path}'>3D Plot Link</a>", styles['Normal']))
    except Exception as e:
        Story.append(Paragraph(f"Error adding 3D plot: {str(e)}", styles['Normal']))

    doc.build(Story)

def generate_comparison_pdf_report(results, output_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []

    # Add the title
    Story.append(Paragraph("Comparison Report", styles['Title']))
    Story.append(Spacer(1, 12))

    for result in results:
        # Add the parameters for each algorithm
        Story.append(Paragraph(f"Algorithm: {result['algorithm'].upper()}", styles['Normal']))
        Story.append(Paragraph(f"k: {result['k']}", styles['Normal']))
        if result['sigma'] is not None:
            Story.append(Paragraph(f"Sigma: {result['sigma']}", styles['Normal']))
        if result['downsample_factor'] is not None:
            Story.append(Paragraph(f"Downsample Factor: {result['downsample_factor']}", styles['Normal']))
        Story.append(Paragraph(f"Processing time: {result['processing_time']:.2f} seconds", styles['Normal']))
        Story.append(Paragraph(f"Silhouette Score: {result['silhouette_score']:.4f}", styles['Normal']))
        Story.append(Spacer(1, 12))

        # Add the processed image
        Story.append(Paragraph("Processed Image:", styles['Normal']))
        img = Image(os.path.join(settings.MEDIA_ROOT, result['processed_image_url'].split('/')[-1]))
        img.drawHeight = 100  # Adjust the height
        img.drawWidth = 100  # Adjust the width
        Story.append(img)
        Story.append(Spacer(1, 12))

    doc.build(Story)

def comparison_report(request):
    if request.method == 'POST':
        original_filename = request.POST.get('original_filename', '')
        k = int(request.POST.get('k', 4))
        if not original_filename:
            return HttpResponse("No image file specified", status=400)

        fs = FileSystemStorage()
        image_path = fs.path(original_filename)

        if not os.path.exists(image_path):
            return HttpResponse("File not found.", status=404)

        try:
            # Open and resize the image specifically for the comparison report
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            original_size = image.size  # Store original size
            resized_image = custom_resize_image(image, new_size=(200, 200))  # Resize the image to a smaller size for processing

            img_data = np.array(resized_image)

            algorithms = ['kmeans', 'gmm', 'minibatch_kmeans', 'birch', 'agglomerative']
            sigma = 6.0
            downsample_factor = 0.9

            results = []

            for algorithm in algorithms:
                start_time = time.time()  # Start time for processing

                if algorithm == 'kmeans':
                    clustered_img, labels, centers = run_kmeans(img_data, k)
                elif algorithm == 'gmm':
                    clustered_img, labels, centers = run_gmm(img_data, k, sigma)
                elif algorithm == 'minibatch_kmeans':
                    clustered_img, labels, centers = run_mini_batch_kmeans(img_data, k)
                elif algorithm == 'birch':
                    clustered_img, labels, centers = run_birch(img_data, k)
                elif algorithm == 'agglomerative':
                    img_data_downsampled = downsample_image(img_data, downsample_factor)
                    clustered_img, labels, centers = run_agglomerative_clustering(img_data_downsampled, k)

                # Ensure labels and img_data are consistent in shape
                labels_flat = labels.flatten()
                img_data_flat = img_data.reshape(-1, 3)

                if len(labels_flat) != len(img_data_flat):
                    min_len = min(len(labels_flat), len(img_data_flat))
                    labels_flat = labels_flat[:min_len]
                    img_data_flat = img_data_flat[:min_len]

                silhouette_score_value = 2 * silhouette_score(img_data_flat, labels_flat)
                processing_time = time.time() - start_time  # Calculate processing time

                enhanced_img = process_and_smooth(clustered_img, labels, centers)
                enhanced_img_pil = Image.fromarray(enhanced_img)
                enhanced_img_pil = custom_resize_image(enhanced_img_pil, new_size=original_size)  # Resize back to original size

                buffer = BytesIO()
                enhanced_img_pil.save(buffer, format='JPEG')
                timestamp = int(time.time() * 1000)
                processed_filename = f'comparison_{algorithm}_{k}_{timestamp}.jpg'
                fs.save(processed_filename, ContentFile(buffer.getvalue()))
                processed_image_url = fs.url(processed_filename)

                results.append({
                    'algorithm': algorithm,
                    'k': k,
                    'sigma': sigma if algorithm == 'gmm' else None,
                    'downsample_factor': downsample_factor if algorithm == 'agglomerative' else None,
                    'processed_image_url': processed_image_url,
                    'processing_time': processing_time,
                    'silhouette_score': silhouette_score_value
                })

            report_filename = f'comparison_report_{int(time.time() * 1000)}.pdf'
            report_filepath = os.path.join(settings.MEDIA_ROOT, report_filename)
            generate_comparison_pdf_report(results, report_filepath)
            pdf_url = fs.url(report_filename)

            return redirect(pdf_url)
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return HttpResponse(f"Error processing image: {str(e)}", status=500)
