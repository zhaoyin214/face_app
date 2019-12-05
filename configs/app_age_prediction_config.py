import os

static_dir = "static"
app_config_name = "faces.age_prediction"
app_name = app_config_name.split(".")[ - 1]
base_dir = os.path.join("images", *app_config_name.split("."))
img_dir = os.path.join(base_dir, "img")
upload_dir = os.path.join(base_dir, "upload")
output_dir = os.path.join(base_dir, "output")


PRETRAINED_MODEL = {
    "proto": "./age_prediction/models/age_deploy.prototxt",
    "weights": "./age_prediction/models/age_net.caffemodel",
    "backend": "caffe",
    "input_height": 227,
    "input_width": 227,
    "swap_rb": False,
    "crop": True,
    "mean": [78.4263377603, 87.7689143744, 114.895847746],
    "scale_factor": 1.0
}

age_list = [
    "(0 - 2)", "(4 - 6)", "(8 - 12)", "(15 - 20)", "(25 - 32)", "(38 - 43)",
    "(48 - 53)", "(60 - 100)"
]

padding = 20
