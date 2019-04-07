from pipelines.inference.detect_api import load_model, infer
import os

BASEDIR = os.path.dirname(os.path.dirname(__file__))

def _test101():

    checkpoint_path = os.path.join(BASEDIR, 'detection_ctpn', 'checkpoints_mlt')
    model = load_model(checkpoint_path)

    img_fn = os.path.join(BASEDIR, 'data', 'text_small.jpg')
    res = infer(img_fn, model)
    print(res)

    return res

if __name__ == '__main__':

    _test101()