import argparse
import json
import cv2
from flask import Flask, render_template, redirect, send_file, url_for, request, jsonify, Response
import os
import numpy as np
import supervision as sv

from ultralytics import YOLO

model = YOLO('model/best-v2.pt')

# model = YOLO('runs/detect/train/weights/best.pt')
app = Flask(__name__)


@app.route("/")
def hello_word():
    with open('static/constants/helpline.json') as f:
        data = json.load(f)
    # return render_template('result.html')
    return render_template('index.html', json_data=data)


@app.route('/', methods=['POST'])
def upload_data(result_image_path=None, speed=None):
    if 'file' in request.files:
        image_file = request.files['file']
        if not image_file.filename:
            return redirect(url_for('hello_word'))
        else:
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', image_file.filename)
            image_file.save(filepath)

            file_extension = image_file.filename.rsplit('.', 1)[1].lower()
            if file_extension == 'mp4':

                video_path = filepath
                print("video path", video_path)
                cap = cv2.VideoCapture(video_path)

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    result = model(frame, save=True)
                    cv2.waitKey(1)

                    res_plotted = result[0].plot()
                    out.write(res_plotted)

                    if cv2.waitKey(1) == ord('q'):
                        break
                out.release()
                cap.release()
                cv2.destroyAllWindows()

                # Download the file
                send_file('output.mp4', as_attachment=True)
                return redirect(url_for('hello_word'))
            else:
                # img = cv2.imread(filepath)
                # frame = cv2.imencode('.'+file_extension, cv2.UMat(img))[1].tobytes()
                filepath_new = f"uploads/{image_file.filename}"
                detections = model.predict(filepath_new, save=True, project="static/output", name="predict",conf=0.3)
                print('test', detections)
                for res in detections:
                    result_image_path = res.save_dir
                    speed = res.speed
                return render_template('result.html', result_speed=speed,
                                       result_image_path="../" + result_image_path + "/" + image_file.filename)
    else:
        print("nothing")
        return 'No image provided.'



@app.route("/detect_objects", methods=['GET'])
def detectObject():
    cap = cv2.VideoCapture(0)
    box_annotator= sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    while True:
        _,frame = cap.read()
        
        result = model(frame)[0]
        detection= sv.Detections.from_ultralytics(result)
        
        frame= box_annotator.annotate(scene=frame,detections=detection)
        cv2.imshow('Camera', frame)
        

        # Check for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    # cv2.destroyAllWindows()
    # results = model.predict(source=0, show=True,stream=True)
    return "hey"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial Object Detection")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
