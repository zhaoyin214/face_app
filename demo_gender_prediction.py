from configs.app_face_detection_config import PRETRAINED_MODEL_RETAIL_0044, THRESHOLD
from configs.app_gender_prediction_config import PRETRAINED_MODEL as PRETRAINED_MODEL_GENDER
from face_detection import face_detector
from gender_prediction import gender_predictor

import cv2

if __name__ == "__main__":

    output_path = "./output/out.jpg"
    image = cv2.imread(filename="./img/13f44c7a-fde1-4a73-9a73-aba5f374f7f3.jpg")

    image_marked, face_bboxes, t_elapsed = face_detector(
        image=image, pretrained_model=PRETRAINED_MODEL_RETAIL_0044, threshold=THRESHOLD,
        output_path=output_path
    )
    print("face bboxes: ", face_bboxes)
    print("elapsed time: ", t_elapsed)

    image_marked, geners, t_elapsed = gender_predictor(
        image=image_marked, pretrained_model=PRETRAINED_MODEL_GENDER,
        bboxes=face_bboxes, output_path=output_path
    )

    print("genders: ", geners)
    print("elapsed time: ", t_elapsed)

    cv2.imshow("gender", image_marked)
    cv2.waitKey(0)
