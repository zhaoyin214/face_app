from configs.app_face_detection_config import PRETRAINED_MODEL_RES10, PRETRAINED_MODEL_OPENCV, THRESHOLD
from configs.app_age_prediction_config import PRETRAINED_MODEL as PRETRAINED_MODEL_AGE
from face_detection import face_detector
from age_prediction import age_predictor

import cv2

if __name__ == "__main__":

    output_path = "./output/out.jpg"
    image = cv2.imread(filename="./img/amanda_bynes.jpg")

    image_marked, face_bboxes, t_elapsed = face_detector(
        image=image, pretrained_model=PRETRAINED_MODEL_RES10, threshold=THRESHOLD,
        output_path=output_path
    )
    print("face bboxes: ", face_bboxes)
    print("elapsed time: ", t_elapsed)

    image_marked, ages, t_elapsed = age_predictor(
        image=image_marked, pretrained_model=PRETRAINED_MODEL_AGE,
        bboxes=face_bboxes, output_path=output_path
    )

    print("ages: ", ages)
    print("elapsed time: ", t_elapsed)

    cv2.imshow("age", image_marked)
    cv2.waitKey(0)
