from flask import request, Flask, jsonify
import numpy as np
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from flask import Flask, request, jsonify
import torch
from PIL import Image, ImageDraw
import numpy as np

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='weightBest.pt')  # local model
print("loaded  model")


@app.route("/detect", methods=["POST"])
def detect():
    """
        Handler of /detect POST endpoint
        Receives uploaded file with a name "image_file", passes it
        through YOLOv8 object detection network and returns and array
        of bounding boxes.
        :return: a JSON array of objects bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    if "image_file" not in request.files:
        return "No 'image_file' in request", 400

    image_file = request.files["image_file"]
    if image_file.filename == "":
        return "No selected file", 400

    # Call your object detection function with the image stream
    # cv2.imwrite('E:/wyr.ai/api_1/output/before.jpg', image_file.stream)

    boxes = detect_objects_on_image(image_file.stream)
   
    return jsonify(boxes)

def detect_objects_on_image(buf, confidence_threshold=0.45):

    # Load the image
    image = Image.open(buf)

    # Extract EXIF data
    exif_data = image._getexif()
    if exif_data is not None:
        # Check for the orientation tag (key 274)
        if 274 in exif_data:
            orientation = exif_data[274]
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)

    # image = image.rotate(0, expand=True)  # This line might not be necessary
    # results = model(image)
    image_np = np.array(image)
    results = model(image_np)
    tensor = results.xyxy[0]
    output = []

    for box in tensor:
        x1 , y1, x2, y2, confidence, classID = box[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        confidence = round(confidence.item(), 2)
        # print(box)
        center_x = int((x1+x2)/2)
        center_y = int((y1+y2)/2)
        output.append([int(classID.item()), center_x, center_y])
        # output.append([center_x, center_y, classID.item(), confidence])
    # Save the image with bounding boxes to the specified folder
    output.sort(key=lambda x: x[1])  # Sort by the first element (x-axis center)
    # cv2.imwrite("output/output_image2.jpg", image_np)
    classes_after_sort = [classID for classID, _, _ in output]
    print(classes_after_sort)
    return classes_after_sort

    image = Image.open(image_stream)
        # Extract EXIF data
    exif_data = image._getexif()
    if exif_data is not None:
        # Check for the orientation tag (key 274)
        if 274 in exif_data:
            orientation = exif_data[274]
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)

    draw = ImageDraw.Draw(image)

    for box in boxes:
        x1, y1, x2, y2, class_name, prob, center_x, center_y = box

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red")

        # Draw center point
        draw.point((center_x, center_y), fill="red")

    return image
if __name__ == '__main__':
    app.run(debug=True)
    # serve(app, port=5000)
