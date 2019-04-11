from pipelines.inference.ocr_api import OcrAPI
import os
import cv2

BASEDIR = os.path.dirname(os.path.dirname(__file__))

def _test101():

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

    ocr = OcrAPI(model_info)

    ocr.load_model()


    img_fn = os.path.join(BASEDIR, 'data', 'bad_case.jpg')
    image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
    for i in range(1):
        res = ocr.infer(image, debug=True)
    print(res)

    return res

    # reco()

if __name__ == '__main__':

    _test101()