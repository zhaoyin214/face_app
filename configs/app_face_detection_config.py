import os

static_dir = "static"
app_config_name = "faces.face_detection"
app_name = app_config_name.split(".")[-1]
base_dir = os.path.join("images", *app_config_name.split("."))
img_dir = os.path.join(base_dir, "img")
upload_dir = os.path.join(base_dir, "upload")
output_dir = os.path.join(base_dir, "output")


PRETRAINED_MODEL_RES10 = {
    "proto": "./face_detection/models/res10_300x300_ssd_deploy.prototxt",
    "weights": "./face_detection/models/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    "backend": "caffe",
    "input_height": 300,
    "input_width": 300,
    "swap_rb": True,
    "crop": False,
    "mean": [104, 117, 123],
    "scale_factor": 1.0
}

PRETRAINED_MODEL_OPENCV = {
    "proto": "./face_detection/models/opencv_face_detector.pbtxt",
    "weights": "./face_detection/models/opencv_face_detector_uint8.pb",
    "backend": "tf",
    "input_height": 300,
    "input_width": 300,
    "swap_rb": True,
    "crop": False,
    "mean": [104, 117, 123],
    "scale_factor": 1.0
}

PRETRAINED_MODEL_RETAIL_0044 = {
    "proto": "./face_detection/models/face-detection-retail-0044.prototxt",
    "weights": "./face_detection/models/face-detection-retail-0044.caffemodel",
    "backend": "caffe",
    # "input_height": 300,
    # "input_width": 300,
    "swap_rb": False,
    "crop": False,
    # "mean": [104, 117, 123],
    # "scale_factor": 1.0
}

PRETRAINED_MODEL_RETAIL_0005_FP32 = {
    "proto": "./face_detection/models/face-detection-retail-0005/FP32/face-detection-retail-0005.xml",
    "weights": "./face_detection/models/face-detection-retail-0005/FP32/face-detection-retail-0005.bin",
    "backend": "intel",
    # "input_height": 300,
    # "input_width": 300,
    "swap_rb": False,
    "crop": False,
    # "mean": [104, 117, 123],
    # "scale_factor": 1.0
}

THRESHOLD = 0.88

