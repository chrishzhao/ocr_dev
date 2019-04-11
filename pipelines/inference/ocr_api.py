from .inference_api import InferenceAPI
from .detect_api import DetectionAPI
from .recognize_api import RecognitionAPI
import os
import cv2
import uuid
import re
from .preprocess import resize_image
import time

class OcrAPI(InferenceAPI):

    def __init__(self, model_info):

        self.detector_model_info = model_info['detector']
        self.recoginzer_model_info = model_info['recognizer']

        self.detector = DetectionAPI(self.detector_model_info)
        self.recoginzer = RecognitionAPI(self.recoginzer_model_info)

    def load_model(self):

        self.detector.load_model()
        self.recoginzer.load_model()

    def remove_noise(self, text):

        pattern = '[\[\]_]'
        return re.sub(pattern, '', text)

    def infer(self, image, debug = False):

        start = time.time()
        img, _, im_fn = resize_image(image, debug)

        boxes = self.detector.infer(img, debug)

        res = []

        for bbox in boxes:
            crop_image, image_id = self.crop(img, bbox, debug)
            text = self.recoginzer.infer(crop_image)
            text = self.remove_noise(text)

            res.append(
                {
                    'bbox': self.bbox2dict(bbox),
                    'text': text,
                    'id': image_id
                }
            )

        res_final = {
            'result': res,
            'resized_im': im_fn
        }

        cost_time = (time.time() - start)
        print("total time: {:.2f}s".format(cost_time))

        return res_final

    def bbox2dict(self, bbox):

        return dict(
            x0 = int(bbox[0]),
            y0 = int(bbox[1]),
            x1 = int(bbox[2]),
            y1 = int(bbox[3]),
            x2 = int(bbox[4]),
            y2 = int(bbox[5]),
            x3 = int(bbox[6]),
            y3 = int(bbox[7])
        )

    def crop(self, image, bbox, debug = False):

        x0 = int(bbox[0])
        y0 = int(bbox[1])
        x1 = int(bbox[2])
        y1 = int(bbox[3])
        x2 = int(bbox[4])
        y2 = int(bbox[5])
        x3 = int(bbox[6])
        y3 = int(bbox[7])

        xmin = min(x0, x1, x2, x3)
        xmax = max(x0, x1, x2, x3)
        ymin = min(y0, y1, y2, y3)
        ymax = max(y0, y1, y2, y3)

        crop_img = image[ymin:ymax, xmin:xmax]

        image_id = str(uuid.uuid4())
        # output_path = os.path.join(
        #     os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        #     'output', 'detection')
        output_path = '/tmp'
        basename = '{}.jpg'.format(image_id)
        # cv2.imwrite(os.path.join(output_path, basename), crop_img[:, :, ::-1])
        if debug and False:
            cv2.imwrite(os.path.join(output_path, basename), crop_img)
        return crop_img, image_id


