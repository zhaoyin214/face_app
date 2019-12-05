#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   age_prediction.py
@time    :   2019/12/04 22:17:15
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   age prediction with opencv dnn module
"""

__author__ = "XiaoY"


import cv2
import time
import numpy as np

from utils.load_net import Net

class AgePredictor(Net):
    """
    age prediction
    """

    def __init__(self, pretrained_model):

        super(AgePredictor, self).__init__(pretrained_model)

        self._age_list = pretrained_model.get("age_list")
        self._padding = pretrained_model.get("padding")
        if not self._age_list:
            self._age_list = [
                "(0 - 2)", "(4 - 6)", "(8 - 12)", "(15 - 20)", "(25 - 32)",
                "(38 - 43)", "(48 - 53)", "(60 - 100)"
            ]
        if not self._padding:
            self._padding = 20

    def predict(self, image, bboxes, output_path):

        image_height, image_width = image.shape[0 : 2]

        t_net = time.time()
        ages = []

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
            age = self._age_list[np.argmax(output[0])]
            ages.append(age)
            print("Age Output : {}".format(output))
            print("Age : {}, conf = {:.3f}".format(age, np.max(output[0])))

            cv2.putText(
                image, age, (bbox[0], bbox[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                cv2.LINE_AA
            )

        cv2.imwrite(output_path, image)

        t_net = time.time() - t_net
        print("time : {:.3f}".format(t_net))

        return image, ages, t_net

    def __call__(self, image, bboxes, output_path):
        return self.predict(
            image=image, bboxes=bboxes, output_path=output_path
        )

def age_predictor(image, pretrained_model, bboxes, output_path):

    t_total = time.time()
    predictor = AgePredictor(pretrained_model=pretrained_model)
    image, ages, t_net = predictor(
        image=image, bboxes=bboxes, output_path=output_path
    )
    t_total = time.time() - t_total

    elapsed_times = (t_net, t_total)

    return image, ages, elapsed_times

