import os
import time

import cv2
import numpy as np
import tensorflow as tf

from detection_ctpn.nets import model_train as model
from detection_ctpn.utils.rpn_msr.proposal_layer import proposal_layer
from detection_ctpn.utils.text_connector.detectors import TextDetector
from detection_ctpn.utils.prepare.resize import resize_image

from .inference_api import InferenceAPI
import uuid

class DetectionAPI(InferenceAPI):

    def __init__(self, model_info):

        self.checkpoint_path = model_info['checkpoint_path']

    def load_model( self ):

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
            input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            bbox_pred, cls_pred, cls_prob = model.model(input_image)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
            sess_config.gpu_options.allow_growth = True

            sess = tf.Session(config=sess_config)
            ckpt_state = tf.train.get_checkpoint_state(self.checkpoint_path)
            model_path = os.path.join(self.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            self.sess = sess
            self.input_image = input_image
            self.input_im_info = input_im_info
            self.bbox_pred = bbox_pred
            self.cls_pred = cls_pred
            self.cls_prob= cls_prob

        self.detection_graph = detection_graph

    def infer( self, img, debug = True ):

        with self.detection_graph.as_default():
            print('===============')
            #print(im_fn)
            start = time.time()
            # try:
            #     im = cv2.imread(im_fn)[:, :, ::-1]
            # except:
            #     raise("Error reading image {}!".format(im_fn))

            h, w, c = img.shape
            im_info = np.array([h, w, c]).reshape([1, 3])

            #sess, input_image, input_im_info, bbox_pred, cls_pred, cls_prob = model

            sess = self.sess
            input_image = self.input_image
            input_im_info = self.input_im_info
            bbox_pred = self.bbox_pred
            cls_pred = self.cls_pred
            cls_prob = self.cls_prob

            bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                   feed_dict={input_image: [img],
                                                              input_im_info: im_info})

            textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
            scores = textsegs[:, 0]
            textsegs = textsegs[:, 1:5]

            textdetector = TextDetector(DETECT_MODE='H')
            boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
            boxes = np.array(boxes, dtype=np.int)

            cost_time = (time.time() - start)
            print("detector cost time: {:.2f}s".format(cost_time))

            if debug:
                for i, box in enumerate(boxes):
                    cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                                  thickness=2)
                #img_r = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)

                # output_path = os.path.join(
                #     os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                #     'output', 'detection')
                output_path = '/tmp'
                basename = '{}.jpg'.format(str(uuid.uuid4()))
                cv2.imwrite(os.path.join(output_path, basename), img[:, :, ::-1])

                with open(os.path.join(output_path, os.path.splitext(basename)[0]) + ".txt",
                          "w") as f:
                    for i, box in enumerate(boxes):
                        line = ",".join(str(box[k]) for k in range(8))
                        line += "," + str(scores[i]) + "\r\n"
                        f.writelines(line)

        return boxes
