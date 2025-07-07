from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Путь к директории для обработки изображений
PROCESSING_DIR = 'static/processed'

# Загрузка модели
model = YOLO('yolov8n.pt')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Сохранение загруженного файла
    image_path = os.path.join(PROCESSING_DIR, file.filename)
    file.save(image_path)

    # Обработка картинки
    process_image(image_path)

    return send_from_directory(PROCESSING_DIR, os.path.basename(image_path).replace('.jpg', '_yolo.jpg'))


def process_image(image_path):
    image = cv2.imread(image_path)
    results = model(image)[0]

    orig_image = results.orig_img
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)

    grouped_objects = {}

    for class_id, box in zip(classes, boxes):
        class_name = classes_names[int(class_id)]
        color = (255, 0, 0)  # Цвет рамки

        if class_name not in grouped_objects:
            grouped_objects[class_name] = []
        grouped_objects[class_name].append(box)

        x1, y1, x2, y2 = box
        cv2.rectangle(orig_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(orig_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Сохранение измененного изображения
    new_image_path = os.path.splitext(image_path)[0] + '_yolo.jpg'
    cv2.imwrite(new_image_path, orig_image)


if __name__ == '__main__':
    # Создание директории, если её нет
    if not os.path.exists(PROCESSING_DIR):
        os.makedirs(PROCESSING_DIR)

    app.run(debug=True)



