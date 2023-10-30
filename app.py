import torch
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image

app = Flask(__name)

# Load the YOLOv8 model using PyTorch
model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s', pretrained=False, path_or_model='/path/to/best.pt')
model.eval()
print("Loaded YOLOv8 model")

@app.route("/detect", methods=["POST"])
def detect():
    # Your detection code here
    if "image_file" not in request.files:
        return "No 'image_file' in request", 400

    image_file = request.files["image_file"]
    if image_file.filename == "":
        return "No selected file", 400

    # Call your object detection function with the image stream
    boxes = detect_objects_on_image(image_file.stream)
    
    # Return the detected bounding boxes as JSON
    return jsonify(boxes)


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    sorted by the x-coordinate of the center point and y-coordinate.
    :param buf: Input image file stream
    :return: Array of bounding boxes in format [[x1, y1, x2, y2, object_type, probability],..]
    """
    # model = YOLO("best.pt")
    results = model.predict(Image.open(buf), conf=0.60)
    print(results)
    result = results[0]
    output = []

    # Create a list of objects with their x and y coordinates
    objects_with_x_and_y = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        x_center = (x1 + x2) / 2  # Calculate the x-coordinate of the center
        y_center = (y1 + y2) / 2  # Calculate the y-coordinate of the center
        objects_with_x_and_y.append([
            x_center, y_center, x1, y1, x2, y2, result.names[class_id], prob
        ])

    # Sort objects by both x-coordinate and y-coordinate
    sorted_objects = sorted(objects_with_x_and_y, key=lambda obj: obj[0])
    sorted_objects = sorted(sorted_objects, key=lambda obj: obj[1])

    # Extract classes from the sorted objects
    classes_only = [obj[6] for obj in sorted_objects]

    return classes_only




if __name__ == '__main__':
    app.run(debug=True)
