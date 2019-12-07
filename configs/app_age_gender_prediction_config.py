import os

static_dir = "static"
app_config_name = "faces.age_gender_prediction"
app_name = app_config_name.split(".")[-1]
base_dir = os.path.join("images", *app_config_name.split("."))
img_dir = os.path.join(base_dir, "img")
upload_dir = os.path.join(base_dir, "upload")
output_dir = os.path.join(base_dir, "output")


PRETRAINED_MODEL = {
    "proto": "./age_gender_prediction/models/age-gender-recognition-retail-0013-fp32.xml",
    "weights": "./age_gender_prediction/models/age-gender-recognition-retail-0013-fp32.bin",
    "backend": "intel",
    "input_height": 62,
    "input_width": 62,
    "swap_rb": False,
    "crop": False,
    "mean": [0, 0, 0],
    "scale_factor": 1.0,
    "padding": 30,
    "age_layer": "age_conv3",
    "age_scale": 100,
    "gender_layer": "prob",
    "gender_list": ["Female", "Male"],
}
