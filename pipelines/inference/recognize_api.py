import tensorflow as tf
import cv2
import numpy as np

from recognition_crnn.config import global_config
from recognition_crnn.crnn_model import crnn_net
from recognition_crnn.data_provider import tf_io_pipline_fast_tools

CFG = global_config.cfg

from .inference_api import InferenceAPI
import time

class RecognitionAPI(InferenceAPI):

    def __init__(self, model_info):

        self.weights_path = model_info['weights_path']
        self.char_dict_path = model_info['char_dict_path']
        self.ord_map_dict_path = model_info['ord_map_dict_path']


    def load_model(self):

        recognition_graph = tf.Graph()
        with recognition_graph.as_default():
            inputdata = tf.placeholder(
                dtype=tf.float32,
                shape=[1, None, None, CFG.ARCH.INPUT_CHANNELS],
                name='input'
            )

            seq_len = tf.placeholder(
                dtype=tf.int32,
                shape=[1],
                name='seq_len'
            )

            codec = tf_io_pipline_fast_tools.CrnnFeatureReader(
                char_dict_path=self.char_dict_path,
                ord_map_dict_path=self.ord_map_dict_path
            )

            net = crnn_net.ShadowNet(
                phase='test',
                hidden_nums=CFG.ARCH.HIDDEN_UNITS,
                layers_nums=CFG.ARCH.HIDDEN_LAYERS,
                num_classes=CFG.ARCH.NUM_CLASSES
            )

            inference_ret = net.inference(
                inputdata=inputdata,
                name='shadow_net',
                reuse=False
            )

            decodes, _ = tf.nn.ctc_beam_search_decoder(
                inputs=inference_ret,
                # sequence_length=CFG.ARCH.SEQ_LENGTH * np.ones(1),
                sequence_length=seq_len,
                beam_width=1,
                merge_repeated=False
            )

            # config tf saver
            saver = tf.train.Saver()

            # config tf session
            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
            sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH

            sess = tf.Session(config=sess_config)

            saver.restore(sess=sess, save_path=self.weights_path)

            self.sess = sess
            self.inference_ret = inference_ret
            self.decodes = decodes
            self.inputdata = inputdata
            self.seq_len = seq_len
            self.codec = codec

        self.recognition_graph = recognition_graph

    def infer( self, image ):

        with self.recognition_graph.as_default():
            start = time.time()
            new_heigth = 32
            scale_rate = new_heigth / image.shape[0]
            new_width = int(scale_rate * image.shape[1])
            new_width = new_width if new_width > 100 else 100
            image = cv2.resize(image, (new_width, new_heigth), interpolation=cv2.INTER_LINEAR)
            image = np.array(image, np.float32) / 127.5 - 1.0

            sess = self.sess
            inference_ret = self.inference_ret
            decodes = self.decodes
            inputdata = self.inputdata
            seq_len = self.seq_len
            codec = self.codec

            ret = sess.run(inference_ret, feed_dict={inputdata: [image], seq_len: [int(new_width/4)]})
            #print(ret.shape)

            preds = sess.run(decodes, feed_dict={inputdata: [image], seq_len: [int(new_width/4)]})

            #print(preds[0])

            preds = codec.sparse_tensor_to_str(preds[0])

            #print('Predict image result {:s}'.format(
            #     preds[0])
            #)
            cost_time = (time.time() - start)
            print("recognizer cost time: {:.2f}s".format(cost_time))

        return preds[0]