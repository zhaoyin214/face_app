#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   gender_prediction.py
@time    :   2019/12/04 16:55:47
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   gender prediction with opencv dnn module
"""

__author__ = "XiaoY"

import cv2
import time
import numpy as np

from utils.load_net import Net

class GenderPredictor(Net):
    """
    gender prediction
    """

    def __init__(self, pretrained_model):

        super(GenderPredictor, self).__init__(pretrained_model)
        # if pretrained_model["backend"].lower() == "tf":
        #     self._net = cv2.dnn.readNetFromTensorflow(
        #         config=pretrained_model["proto"],
        #         model=pretrained_model["weights"]
        #     )
        # elif pretrained_model["backend"].lower() == "caffe":
        #     self._net = cv2.dnn.readNetFromCaffe(
        #         prototxt=pretrained_model["proto"],
        #         caffeModel=pretrained_model["weights"]
        #     )

        # self._input_height = pretrained_model.get("input_height")
        # self._input_width = pretrained_model.get("input_width")
        # self._swap_rb = pretrained_model.get("swap_rb")
        # self._crop = pretrained_model.get("crop")
        # self._mean = pretrained_model.get("mean")
        # self._scale_factor = pretrained_model.get("scale_factor")

        # if not self._swap_rb:
        #     self._swap_rb = False
        # if not self._crop:
        #     self._crop = False
        # if not self._mean:
        #     self._mean = (0, 0, 0)
        # if not self._scale_factor:
        #     self._scale_factor = 1 / 255

        self._gender_list = pretrained_model.get("gender_list")
        self._padding = pretrained_model.get("padding")
        if not self._gender_list:
            self._gender_list = ["Male", "Female"]
        if not self._padding:
            self._padding = 20

    def predict(self, image, bboxes, output_path):

        image_height, image_width = image.shape[0 : 2]

        t_net = time.time()
        genders = []

        for bbox in bboxes:

            roi = image[
                np.max(
                    [0, bbox[1] - self._padding]
                ) : np.min(
                    [bbox[3] + self._padding, image_height - 1]
                ),
                np.max(
                    [0, bbox[0] - self._padding]
                ) : np.min(
                    [bbox[2] + self._padding, image_width - 1]
                ), :
            ]

            if not roi.size:
                continue

            input_blob = cv2.dnn.blobFromImage(
                image=roi, scalefactor=self._scale_factor,
                size=(self._input_width, self._input_height),
                mean=self._mean, swapRB=self._swap_rb, crop=self._crop
            )
            self._net.setInput(input_blob)
            output = self._net.forward()
            gender = self._gender_list[np.argmax(output[0])]
            genders.append(gender)
            print("Gender Output : {}".format(output))
            print("Gender : {}, conf = {:.3f}".format(gender, np.max(output[0])))

            cv2.putText(
                image, gender, (bbox[0], bbox[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                cv2.LINE_AA
            )

        cv2.imwrite(output_path, image)

        t_net = time.time() - t_net
        print("time : {:.3f}".format(t_net))

        return image, genders, t_net

    def __call__(self, image, bboxes, output_path):
        return self.predict(
            image=image, bboxes=bboxes, output_path=output_path
        )

def gender_predictor(image, pretrained_model, bboxes, output_path):

    t_total = time.time()
    predictor = GenderPredictor(pretrained_model=pretrained_model)
    image, genders, t_net = predictor(
        image=image, bboxes=bboxes, output_path=output_path
    )
    t_total = time.time() - t_total

    elapsed_times = (t_net, t_total)

    return image, genders, elapsed_times
