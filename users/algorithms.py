import os
from PIL import Image, ImageFilter
from django.conf import settings
import numpy as np
from sklearn.cluster import  KMeans, MiniBatchKMeans, Birch, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.filters import gaussian
import plotly.graph_objs as go

def run_kmeans(image_data, n_clusters):
    pixels = image_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    clustered_img = centers[labels].reshape(image_data.shape).astype(np.uint8)
    return clustered_img, labels, centers

def run_gmm(image_data, n_clusters, sigma):
    pixels = image_data.reshape(-1, 3)
    gmm = GaussianMixture(n_clusters, covariance_type='tied', reg_covar=sigma)
    labels = gmm.fit_predict(pixels)
    centers = gmm.means_
    clustered_img = centers[labels].reshape(image_data.shape).astype(np.uint8)
    return clustered_img, labels, centers

def run_mini_batch_kmeans(image_data, n_clusters):
    pixels = image_data.reshape(-1, 3)
    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters)
    labels = mbkmeans.fit_predict(pixels)
    centers = mbkmeans.cluster_centers_
    clustered_img = centers[labels].reshape(image_data.shape).astype(np.uint8)
    return clustered_img, labels, centers

def sample_data(pixels, max_points=10000):
    if len(pixels) > max_points:
        idx = np.random.choice(len(pixels), size=max_points, replace=False)
        return pixels[idx]
    return pixels

def run_birch(image_data, n_clusters, max_points=10000):
    pixels = image_data.reshape(-1, 3)
    sampled_pixels = sample_data(pixels, max_points)
    birch = Birch(n_clusters=n_clusters)
    birch.fit(sampled_pixels)
    labels = birch.predict(pixels)
    centers = np.array([pixels[labels == i].mean(axis=0) for i in range(n_clusters)])
    clustered_img = centers[labels].reshape(image_data.shape).astype(np.uint8)
    return clustered_img, labels, centers

def run_agglomerative_clustering(image_data, n_clusters, max_points=10000):
    pixels = image_data.reshape(-1, 3)
    sampled_pixels = sample_data(pixels, max_points)
    
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels_sampled = clustering.fit_predict(sampled_pixels)

    # Se atribuie etichete pixelilor originali folosind cei mai apropiați vecini
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(sampled_pixels)
    nearest_neighbors = nn.kneighbors(pixels, return_distance=False).flatten()
    labels = labels_sampled[nearest_neighbors]

    centers = np.array([pixels[labels == i].mean(axis=0) for i in range(n_clusters)])
    clustered_img = centers[labels].reshape(image_data.shape).astype(np.uint8)

    return clustered_img, labels, centers

def downsample_image(image_data, factor):
    if factor <= 0 or factor > 1:
        raise ValueError("Downsample factor must be between 0 and 1.")

    # Se calculeaza dimensiuni noi
    height, width, _ = image_data.shape
    new_height = int(height * factor)
    new_width = int(width * factor)

    # Se redimensioneaza imaginea folosind PIL
    image = Image.fromarray(image_data)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return np.array(image)

def process_and_smooth(clustered_img, labels, centers):
    enhanced_img = clustered_img.copy()

    # Se aplica Gaussian smoothing
    enhanced_img = gaussian(enhanced_img, sigma=1, channel_axis=-1)
    
    # Corectie de culori si se schimba tipul imaginii
    enhanced_img = (enhanced_img * 255).astype(np.uint8)
    
    return enhanced_img

def save_plot_2d(pixels, labels, centers, filename_prefix='plot_2d'):
    # Plot pentru Red vs Green
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster = pixels[labels == label]
        cluster_color = tuple(centers[label] / 255)
        plt.scatter(cluster[:, 0], cluster[:, 1], color=cluster_color, label=f'Cluster {label}', alpha=0.5)

    plt.title(f'2D Color Distribution with k={len(centers)} (Red vs Green)')
    plt.xlabel('Channel 1 (Red)')
    plt.ylabel('Channel 2 (Green)')
    plt.legend()

    plot_path_rg = os.path.join(settings.MEDIA_ROOT, f'{filename_prefix}_rg.png')
    plt.savefig(plot_path_rg)
    plt.close()

    # Plot pentru Green vs Blue
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        cluster = pixels[labels == label]
        cluster_color = tuple(centers[label] / 255)
        plt.scatter(cluster[:, 1], cluster[:, 2], color=cluster_color, label=f'Cluster {label}', alpha=0.5)

    plt.title(f'2D Color Distribution with k={len(centers)} (Green vs Blue)')
    plt.xlabel('Channel 1 (Green)')
    plt.ylabel('Channel 2 (Blue)')
    plt.legend()

    plot_path_gb = os.path.join(settings.MEDIA_ROOT, f'{filename_prefix}_gb.png')
    plt.savefig(plot_path_gb)
    plt.close()

    # Plot pentru Blue vs Red
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        cluster = pixels[labels == label]
        cluster_color = tuple(centers[label] / 255)
        plt.scatter(cluster[:, 2], cluster[:, 0], color=cluster_color, label=f'Cluster {label}', alpha=0.5)

    plt.title(f'2D Color Distribution with k={len(centers)} (Blue vs Red)')
    plt.xlabel('Channel 1 (Blue)')
    plt.ylabel('Channel 2 (Red)')
    plt.legend()

    plot_path_br = os.path.join(settings.MEDIA_ROOT, f'{filename_prefix}_br.png')
    plt.savefig(plot_path_br)
    plt.close()

    plot_urls = {
        'rg': f'{settings.MEDIA_URL}{filename_prefix}_rg.png',
        'gb': f'{settings.MEDIA_URL}{filename_prefix}_gb.png',
        'br': f'{settings.MEDIA_URL}{filename_prefix}_br.png',
    }

    for key, value in plot_urls.items():
        print(f"{key} plot URL: {value}")

    return plot_urls

def save_plot_3d(pixels, labels=None, centers=None, filename='plot_3d.html', max_points=5000):
    pixels = np.asarray(pixels)
    if labels is not None:
        labels = np.asarray(labels)

    if len(pixels) > max_points:
        indices = np.random.choice(range(len(pixels)), size=max_points, replace=False)
        pixels = pixels[indices]
        if labels is not None:
            labels = labels[indices]

    if labels is not None and centers is not None:
        colors = centers[labels]
    else:
        colors = pixels

    fig = go.Figure(data=[go.Scatter3d(
        x=pixels[:, 0], y=pixels[:, 1], z=pixels[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=['rgb({},{},{})'.format(int(c[0]), int(c[1]), int(c[2])) for c in colors],
            opacity=0.5,
        )
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title='Red',
            yaxis_title='Green',
            zaxis_title='Blue'
        ),
        width=600,
        height=400
    )
    plot_path = os.path.join(settings.MEDIA_ROOT, filename)
    fig.write_html(plot_path)
    return f'{settings.MEDIA_URL}{filename}'

def crop_image(image, crop_values):
    # Decupare de imagine. Este o tupla de tipul (left, upper, right, lower)
    return image.crop(crop_values)

def resize_image_algo(image, resize_values):
    # Redimensionare de imagine. Este o tupla de tipul (width, height)
    return image.resize(resize_values, Image.Resampling.LANCZOS)

def apply_filter(image, filter_type):
    filters = {
        'BLUR': ImageFilter.BLUR,
        'CONTOUR': ImageFilter.CONTOUR,
        'DETAIL': ImageFilter.DETAIL,
        'EDGE_ENHANCE': ImageFilter.EDGE_ENHANCE,
        'EMBOSS': ImageFilter.EMBOSS,
        'SHARPEN': ImageFilter.SHARPEN
    }
    return image.filter(filters[filter_type])