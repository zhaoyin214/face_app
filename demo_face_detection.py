from configs.app_face_detection_config import \
    PRETRAINED_MODEL_OPENCV, PRETRAINED_MODEL_RES10, \
    PRETRAINED_MODEL_RETAIL_0044, PRETRAINED_MODEL_RETAIL_0005_FP32, \
    THRESHOLD
from face_detection import face_detector

import cv2

if __name__ == "__main__":

    output_path = "./output/out.jpg"
    # image = cv2.imread(filename="./img/190.jpg")
    # image = cv2.imread(filename="./img/29.jpg")
    image = cv2.imread(filename="./img/13f44c7a-fde1-4a73-9a73-aba5f374f7f3.jpg")
    image_marked, face_bboxes, t_elapsed = face_detector(
        image=image, pretrained_model=PRETRAINED_MODEL_RETAIL_0005_FP32, threshold=THRESHOLD,
        output_path=output_path
    )
    print("face bboxes: ", face_bboxes)
    print("elapsed time: ", t_elapsed)

    cv2.imshow("Faces", image_marked)
    cv2.waitKey(0)
