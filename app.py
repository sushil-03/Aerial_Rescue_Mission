import json
import cv2
from flask import Flask, render_template, redirect, send_file, url_for, request
import os
import supervision as sv
from ultralytics import YOLO
from preprocessing_prediction  import preprocess_and_predict

model = YOLO('model/best.pt')
app = Flask(__name__)

@app.route("/")
def hello_word():
    with open('static/constants/helpline.json') as f:
        data = json.load(f)
    return render_template('index.html', json_data= data)


@app.route('/', methods=['POST'])
def upload_data(results=None ,result_speed=None):
    if 'file' in request.files:
        image_file = request.files['file']
        print('image_file',image_file)
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
                filepath_new = f"uploads/{image_file.filename}"
                result_paths, speeds = preprocess_and_predict(model, filepath_new)
                return render_template('result.html',result_speed=speeds,results=result_paths)
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
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return ""


if __name__ == "__main__":
    app.run(debug=True)

