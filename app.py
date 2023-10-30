from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image

app = Flask(__name__)
model = YOLO("best.pt")  # Replace 0.5 with your desired confidence threshold

# @app.route("/")
# def root():
#     """
#     Site main page handler function.
#     :return: Content of index.html file
#     """
#     with open("index.html") as file:
#         return file.read()


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
    boxes = detect_objects_on_image(image_file.stream)
    
    # Return the detected bounding boxes as JSON
    return jsonify(boxes)


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    # model = YOLO("best.pt")
    results = model.predict(Image.open(buf), conf=0.60 )
    print(results)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output



# serve(app, host='0.0.0.0', port=8080)
if __name__ == '__main__':
    app.run(debug=True)
