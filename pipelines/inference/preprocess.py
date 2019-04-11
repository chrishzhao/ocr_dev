import numpy as np
import cv2
import uuid

def resize_image(img, debug = False):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    # im_scale = float(600) / float(im_size_min)
    # if np.round(im_scale * im_size_max) > 1200:
    #     im_scale = float(1200) / float(im_size_max)
    # im_scale = float(2400) / float(im_size_min)
    # if np.round(im_scale * im_size_max) > 4800:
    #     im_scale = float(4800) / float(im_size_max)

    # im_scale = float(2400) / float(im_size_min)
    # if np.round(im_scale * im_size_max) > 4800:
    #     im_scale = float(4800) / float(im_size_max)

    x = 2400

    if float(im_size_min)>x:
        im_scale = float(x) / float(im_size_min)
        if np.round(im_scale * im_size_max) > x * 2:
            im_scale = float(x * 2) / float(im_size_max)
    else:
        im_scale = 1
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    im_id = str(uuid.uuid4())
    im_fn = '/tmp/{}.jpg'.format(im_id)
    if debug:
        cv2.imwrite(im_fn, re_im)
    return re_im, (new_h / img_size[0], new_w / img_size[1]), im_fn