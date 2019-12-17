#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file    :   age_gender_prediction.py
@time    :   2019/12/07 17:51:57
@author  :   XiaoY
@version :   1.0
@contact :   zhaoyin214@qq.com
@license :   (c)copyright XiaoY
@desc    :   age & gender prediction
"""

__author__ = "XiaoY"


import cv2
import time
import numpy as np

from utils.load_net import Net


class AgeGenderPredictor(Net):

    def __init__(self, pretrained_model):

        super(AgeGenderPredictor, self).__init__(pretrained_model)

        self._padding = pretrained_model.get("padding")
        self._age_layer = pretrained_model.get("age_layer")
        self._age_scale = pretrained_model.get("age_scale")
        self._gender_layer = pretrained_model.get("gender_layer")
        self._gender_list = pretrained_model.get("gender_list")

        if not self._padding:
            self._padding = 20

    def predict(self, image, bboxes, output_path):

        # image_height, image_width = image.shape[0 : 2]

        t_net = time.time()
        ages = []
        genders = []

        for bbox in bboxes:

            # roi = image[
            #     np.max(
            #         [0, bbox[1] - self._padding]
            #     ) : np.min(
            #         [bbox[3] + self._padding, image_height - 1]
            #     ),
            #     np.max(
            #         [0, bbox[0] - self._padding]
            #     ) : np.min(
            #         [bbox[2] + self._padding, image_width - 1]
            #     ), :
            # ]
            roi = self._roi(image, bbox)

            if not roi.size:
                continue

            input_blob = cv2.dnn.blobFromImage(
                image=roi, scalefactor=self._scale_factor,
                size=(self._input_width, self._input_height),
                mean=self._mean, swapRB=self._swap_rb, crop=self._crop
            )
            self._net.setInput(input_blob)
            age, gender = self._net.forward(
                [self._age_layer, self._gender_layer]
            )
            age = int(age.ravel()[0] * self._age_scale)
            gender = self._gender_list[np.argmax(gender.ravel())]
            print("Age: {}, Gender: {}".format(age, gender))

            ages.append(age)
            genders.append(gender)

            text = "{}, {}".format(age, gender)
            cv2.putText(
                image, text, (bbox[0], bbox[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                cv2.LINE_AA
            )

        cv2.imwrite(output_path, image)

        t_net = time.time() - t_net
        print("time : {:.3f}".format(t_net))

        return image, ages, genders, t_net

    def _roi(self, image, bbox):

        image_height, image_width = image.shape[0 : 2]
        aspect_ratio = self._input_width / self._input_height
        box_width = bbox[3] - bbox[1] + 1
        box_height = bbox[2] - bbox[0] + 1
        padding_y = int(box_height * self._padding / 2)
        padding_x = int(
            ((box_height + 2 * padding_y) * aspect_ratio - box_width) / 2
        )

        roi = image[
            np.max(
                [0, bbox[1] - padding_y]
            ) : np.min(
                [bbox[3] + 1 + padding_y, image_height]
            ),
            np.max(
                [0, bbox[0] - padding_x]
            ) : np.min(
                [bbox[2] + 1 + padding_x, image_width]
            ), :
        ]

        return roi

    def __call__(self, image, bboxes, output_path):

        return self.predict(
            image=image, bboxes=bboxes, output_path=output_path
        )

def age_gender_predictor(image, pretrained_model, bboxes, output_path):

    t_total = time.time()
    predictor = AgeGenderPredictor(pretrained_model=pretrained_model)
    image, ages, genders, t_net = predictor(
        image=image, bboxes=bboxes, output_path=output_path
    )
    t_total = time.time() - t_total

    elapsed_times = (t_net, t_total)

    return image, ages, genders, elapsed_times

