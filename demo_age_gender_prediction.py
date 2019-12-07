from configs.app_face_detection_config import THRESHOLD, \
    PRETRAINED_MODEL_RETAIL_0005_FP32
from configs.app_age_gender_prediction_config import PRETRAINED_MODEL as PRETRAINED_MODEL_AGE_GENDER
from face_detection import face_detector
from age_gender_prediction import age_gender_predictor

import cv2

if __name__ == "__main__":

    output_path = "./output/out.jpg"
    image = cv2.imread(filename="./img/sample1.jpg")

    image_marked, face_bboxes, t_elapsed = face_detector(
        image=image, pretrained_model=PRETRAINED_MODEL_RETAIL_0005_FP32, threshold=THRESHOLD,
        output_path=output_path
    )
    print("face bboxes: ", face_bboxes)
    print("elapsed time: ", t_elapsed)

    image_marked, ages, genders, t_elapsed = age_gender_predictor(
        image=image_marked, pretrained_model=PRETRAINED_MODEL_AGE_GENDER,
        bboxes=face_bboxes, output_path=output_path
    )

    print("Age: {}, Gender: {}".format(ages, genders))
    print("elapsed time: ", t_elapsed)

    cv2.imshow("gender", image_marked)
    cv2.waitKey(0)
