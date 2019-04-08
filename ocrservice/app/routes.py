from . import app
from flask import request
from pipelines.inference.ocr_api import OcrAPI
import os
import cv2
import json

BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def get_model_info():
    app.logger.debug('basedir: {}'.format(BASEDIR))
    checkpoint_path = os.path.join(BASEDIR, 'detection_ctpn', 'checkpoints_mlt')
    d_model_info = {
        'checkpoint_path': checkpoint_path
    }

    weights_path = os.path.join(BASEDIR, 'recognition_crnn/model/zh_model/shadownet.ckpt')
    char_dict_path = os.path.join(BASEDIR, 'recognition_crnn/data/char_dict/char_dict_cn.json')
    ord_map_dict_path = os.path.join(BASEDIR, 'recognition_crnn/data/char_dict/ord_map_cn.json')

    r_model_info = {
        'weights_path': weights_path,
        'char_dict_path': char_dict_path,
        'ord_map_dict_path': ord_map_dict_path
    }

    model_info = {
        'detector': d_model_info,
        'recognizer': r_model_info
    }
    return model_info

global ocr
ocr = OcrAPI(get_model_info())
ocr.load_model()

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"


@app.route('/ocr', methods=['POST'])
def ocr_service():
    if request.method == 'POST':
        f = request.files['image']
        img_fn = '/tmp/' + f.filename
        f.save(img_fn)
        image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
        res = ocr.infer(image, debug=True)

        return json.dumps(res, ensure_ascii=False)