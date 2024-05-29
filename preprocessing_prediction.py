import os
import cv2
from preprocessing import combined_preprocessing ,histogram_equalization,gaussian_blur,edge_detection,resize_image,normalize_image,clahe # Import your preprocessing functions

def load_image(image_path):
    return cv2.imread(image_path)

def preprocess_and_predict(model, image_path):
    result_paths = []
    speeds = []
    labels=[]

    print(image_path)
    # Load the image
    image = load_image(image_path)

    # Apply preprocessing
    comibne_preprocessed_image_path = combined_preprocessing(image_path)
    histo_image_path=histogram_equalization(image_path)
    gaussian_image_path=gaussian_blur(image_path)
    edge_image_path=edge_detection(image_path)
    resize_image_path=resize_image(image_path)
    normalize_image_path=normalize_image(image_path)
    clahe_image_path=clahe(image_path)


    detections_original = model.predict(image_path, save=True, project="static/output/original", name="predict", conf=0.3)
    
    # Run the model on preprocessed image
    detections_combine = model.predict(comibne_preprocessed_image_path, save=True, project="static/output/preprocessed", name="predict", conf=0.3)
    detections_histo = model.predict(histo_image_path, save=True, project="static/output/preprocessed", name="predict", conf=0.3)
    detections_gaussian = model.predict(gaussian_image_path, save=True, project="static/output/preprocessed", name="predict", conf=0.3)
    detections_edge = model.predict(edge_image_path, save=True, project="static/output/preprocessed", name="predict", conf=0.3)
    detections_resize= model.predict(resize_image_path, save=True, project="static/output/preprocessed", name="predict", conf=0.3)
    detections_normalize = model.predict(normalize_image_path, save=True, project="static/output/preprocessed", name="predict", conf=0.3)
    detections_clahe = model.predict(clahe_image_path, save=True, project="static/output/preprocessed", name="predict", conf=0.3)
     

    for res in detections_original:
        result_image_path_original = res.save_dir
        speed_original = res.speed
        image="../" + result_image_path_original + "/" + os.path.basename(image_path)
        label="Original"
        result_paths.append({'image':image,'label':label})
        speeds.append(speed_original)

    for res in detections_histo:
        result_image_path_preprocessed = res.save_dir
        speed_preprocessed = res.speed
        # result_paths.append("../" + result_image_path_preprocessed + "/histogram_" + os.path.basename(image_path))
        image="../" + result_image_path_preprocessed + "/histogram_" + os.path.basename(image_path)
        label="Histogram Equalization"
        result_paths.append({'image':image,'label':label})
        speeds.append(speed_preprocessed)

    for res in detections_gaussian:
        result_image_path_preprocessed = res.save_dir
        speed_preprocessed = res.speed
        image="../" + result_image_path_preprocessed + "/gaussian_" + os.path.basename(image_path)
        label="Gaussian Blur"
        result_paths.append({'image':image,'label':label})
        speeds.append(speed_preprocessed)

    # for res in detections_edge:
    #     result_image_path_preprocessed = res.save_dir
    #     speed_preprocessed = res.speed
    #     # result_paths.append("../" + result_image_path_preprocessed + "/edge_" + os.path.basename(image_path))

    #     image="../" + result_image_path_preprocessed + "/edge_" + os.path.basename(image_path)
    #     label="Edge Detection"
    #     result_paths.append({'image':image,'label':label})
    #     speeds.append(speed_preprocessed)

    
    for res in detections_resize:
        result_image_path_preprocessed = res.save_dir
        speed_preprocessed = res.speed
        image="../" + result_image_path_preprocessed + "/resize_" + os.path.basename(image_path)
        label="Resize"
        result_paths.append({'image':image,'label':label})
        speeds.append(speed_preprocessed)

    for res in detections_normalize:
        result_image_path_preprocessed = res.save_dir
        speed_preprocessed = res.speed

        image="../" + result_image_path_preprocessed + "/normalize_" + os.path.basename(image_path)
        label="Normalization"
        result_paths.append({'image':image,'label':label})
        speeds.append(speed_preprocessed)
    
    for res in detections_clahe:
          result_image_path_preprocessed = res.save_dir
          speed_preprocessed = res.speed
          image="../" + result_image_path_preprocessed + "/clahe_" + os.path.basename(image_path)
          label="Clahe"
          result_paths.append({'image':image,'label':label})
          speeds.append(speed_preprocessed)
    for res in detections_combine:
          result_image_path_preprocessed = res.save_dir
          speed_preprocessed = res.speed
          image="../" + result_image_path_preprocessed + "/combine_" + os.path.basename(image_path)
          label="Combine processing"
          result_paths.append({'image':image,'label':label})
          speeds.append(speed_preprocessed)
    return result_paths, speeds

