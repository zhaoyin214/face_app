#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   face_detection.py
@time    :   2019/12/02 23:20:27
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   face detection with opencv dnn module
"""

__author__ = "XiaoY"


import cv2
import time
import numpy as np

from utils.load_net import Net

class FaceDetector(Net):
    """
    face box detection
    """

    def __init__(self, pretrained_model):

        super(FaceDetector, self).__init__(pretrained_model)

    def predict(self, image, threshold, output_path):

        image_height, image_width = image.shape[0 : 2]

        if self._input_height and self._input_width:
            input_width = self._input_width
            input_height = self._input_height
        elif self._input_height:
            input_height = self._input_height
            aspect_ratio = image_width / image_height
            # input image dimensions for the network
            input_width = int(aspect_ratio * input_height)
        elif self._input_width:
            input_width = self._input_width
            aspect_ratio = image_width / image_height
            # input image dimensions for the network
            input_height = int(input_width / aspect_ratio)
        else:
            input_height = image_height
            input_width = image_width

        # input image dimensions for the network
        input_blob = cv2.dnn.blobFromImage(
            image=image, scalefactor=self._scale_factor,
            size=(input_width, input_height),
            mean=self._mean, swapRB=self._swap_rb, crop=self._crop
        )

        t_net = time.time()
        self._net.setInput(input_blob)
        output = self._net.forward()
        t_net = time.time() - t_net

        # bounding boxes
        bboxes = []
        for i in range(output.shape[2]):

            # confidence map of corresponding hand's part.
            conf = output[0, 0, i, 2]
            if conf > threshold:
                x_min = int(output[0, 0, i, 3] * image_width)
                y_min = int(output[0, 0, i, 4] * image_height)
                x_max = int(output[0, 0, i, 5] * image_width)
                y_max = int(output[0, 0, i, 6] * image_height)

                bboxes.append([x_min, y_min, x_max, y_max])
                cv2.rectangle(
                    img=image, pt1=(x_min, y_min), pt2=(x_max, y_max),
                    color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA
                )

        cv2.imwrite(output_path, image)

        return image, bboxes, t_net

    def __call__(self, image, threshold, output_path):
        return self.predict(
            image=image, threshold=threshold, output_path=output_path
        )


def face_detector(image, pretrained_model, threshold, output_path):

    t_total = time.time()
    detector = FaceDetector(pretrained_model=pretrained_model)
    image, bboxes, t_net = detector(
        image=image, threshold=threshold, output_path=output_path
    )
    t_total = time.time() - t_total

    elapsed_times = (t_net, t_total)

    return image, bboxes, elapsed_times

