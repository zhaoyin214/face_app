#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   face_deteciton.py
@time    :   2019/12/17 16:46:07
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   face detection by dlib
"""

__author__ = "XiaoY"

import cv2
import time

from dlib import get_frontal_face_detector

class FaceDetector(object):
    """
    face box detection by dlib
    """

    def __init__(self, upsample_num_times=0, adjust_threshold=0.0):
        """
        """
        self._detector = get_frontal_face_detector()
        self._upsample_num_times = upsample_num_times
        self._adjust_threshold = adjust_threshold

    def predict(self, image, output_path):

        t_net = time.time()
        bboxes, scores, _ = self._detector.run(
            image, self._upsample_num_times, self._adjust_threshold
        )
        t_net = time.time() - t_net

        # bounding boxes
        for bbox, score in zip(bboxes, scores):

            cv2.rectangle(
                img=image,
                pt1=(bbox.left(), bbox.bottom()),
                pt2=(bbox.right(), bbox.top()),
                color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA
            )
            cv2.putText(
                img=image, text="{:.3f}".format(score),
                org=(bbox.left(), bbox.top()),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA
            )

        cv2.imwrite(output_path, image)

        return image, bboxes, t_net

    def __call__(self, image, output_path):
        return self.predict(
            image=image, output_path=output_path
        )


def face_detector(image, output_path, upsample_num_times=0):

    t_total = time.time()
    detector = FaceDetector(upsample_num_times=upsample_num_times)
    image, bboxes, t_net = detector(
        image=image, output_path=output_path
    )
    t_total = time.time() - t_total

    elapsed_times = (t_net, t_total)

    return image, bboxes, elapsed_times


if __name__ == "__main__":

    output_path = "./output/out.jpg"
    image = cv2.imread(filename="./img/29.jpg")
    # image = cv2.imread(filename="./img/13f44c7a-fde1-4a73-9a73-aba5f374f7f3.jpg")
    image_marked, face_bboxes, t_elapsed = face_detector(
        image=image, output_path=output_path, upsample_num_times=0
    )
    print("face bboxes: ", face_bboxes)
    print("elapsed time: ", t_elapsed)

    cv2.imshow("Faces", image_marked)
    cv2.waitKey(0)
