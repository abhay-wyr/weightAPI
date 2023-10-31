from ultralytics import YOLO
from flask import request, Flask, jsonify
# from waitress import serve
import numpy as np
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

app = Flask(__name__)
model = YOLO("bestIN.pt")  # Replace 0.5 with your desired confidence threshold
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
    
    # Return the detected bounding boxes as JSON
    return jsonify(boxes)


def detect_objects_on_image(buf, output_folder='E:/wyr.ai/api_1/output', output_image_name='temp.jpg', confidence_threshold=0.45):
    """
    Function receives an image,
    passes it through YOLOv8 neural network,
    returns an array of detected objects,
    and draws bounding boxes on the image to visualize the results.
    :param buf: Input image file stream
    :param output_folder: Path to the output folder
    :param output_image_name: Name of the output image file
    :param confidence_threshold: Confidence threshold for object detection
    """
    ori_image = os.path.join(output_folder, 'ori.jpg')

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

    image = image.rotate(0, expand=True)  # This line might not be necessary

    results = model.predict(image, conf=0.80)
    result = results[0]
    output = []

    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        class_name = result.names[class_id]
        center_x =  int((x1 + x2) / 2)
        center_y =  int((y1 + y2) / 2)
        output.append([
            center_x, center_y, class_name, prob
        ])
        # Draw bounding box (if you need it)
        # ...
        # Draw a point at the center
        # point_color = (0, 0, 255)  # Red
        # point_radius = 5
        # cv2.circle(image, (center_x, center_y), point_radius, point_color, -1)  # -1 means fill the circle

    # Save the image with bounding boxes to the specified folder
    output.sort(key=lambda x: x[0])  # Sort by the first element (x-axis center)

   
    return output



if __name__ == '__main__':
    app.run(debug=True)
    # serve(app, port=5000)
