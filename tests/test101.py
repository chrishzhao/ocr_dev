from pipelines.inference.detect_api import DetectionAPI
import os
import cv2

BASEDIR = os.path.dirname(os.path.dirname(__file__))

def _test101():

    checkpoint_path = os.path.join(BASEDIR, 'detection_ctpn', 'checkpoints_mlt')
    model_info = {
        'checkpoint_path': checkpoint_path
    }

    detector = DetectionAPI(model_info)

    detector.load_model()

    im_fn = os.path.join(BASEDIR, 'data', 'text_small.jpg')
    im = cv2.imread(im_fn)[:, :, ::-1]
    res = detector.infer(im)
    print(res)

    return res

if __name__ == '__main__':

    _test101()