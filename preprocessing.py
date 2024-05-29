# preprocessing.py
import cv2
import numpy as np

def histogram_equalization(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    equalized_image = cv2.equalizeHist(image)
    preprocessed_image_path = 'uploads/histogram_'+image_path.rsplit('/', 1)[1] 

    cv2.imwrite(preprocessed_image_path, equalized_image)
    return preprocessed_image_path

def gaussian_blur(image_path, kernel_size=(5, 5), sigma=0):
    image = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    preprocessed_image_path = 'uploads/gaussian_'+image_path.rsplit('/', 1)[1] 

    cv2.imwrite(preprocessed_image_path, blurred_image)
    return preprocessed_image_path

def edge_detection(image_path, threshold1=100, threshold2=200):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, threshold1, threshold2)
    preprocessed_image_path = 'uploads/edge_'+image_path.rsplit('/', 1)[1] 

    cv2.imwrite(preprocessed_image_path, edges)
    return preprocessed_image_path

def resize_image(image_path, width=950, height=950):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (width, height))
    preprocessed_image_path = 'uploads/resize_'+image_path.rsplit('/', 1)[1] 
    cv2.imwrite(preprocessed_image_path, resized_image)
    return preprocessed_image_path

def normalize_image(image_path):
    image = cv2.imread(image_path)
    normalized_image = image / 255.0
    preprocessed_image_path = 'uploads/normalize_'+image_path.rsplit('/', 1)[1] 

    cv2.imwrite(preprocessed_image_path, (normalized_image * 255).astype(np.uint8))
    return preprocessed_image_path

def clahe(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(image)
    preprocessed_image_path = 'uploads/clahe_'+image_path.rsplit('/', 1)[1] 
    cv2.imwrite(preprocessed_image_path, clahe_image)
    return preprocessed_image_path

def combined_preprocessing(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(blurred_image)
    
 
    preprocessed_image_path = 'uploads/combine_'+image_path.rsplit('/', 1)[1] 
    cv2.imwrite(preprocessed_image_path,clahe_image)
    
    return preprocessed_image_path
