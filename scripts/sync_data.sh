#!/bin/bash

PROJ_PATH=..
cd $PROJ_PATH

# copy model
#rsync -razv recognition_crnn/model/zh_model huasha@hb:ocr_dev/recognition_crnn/model/
#rsync -razv detection_ctpn/checkpoints_mlt huasha@hb:ocr_dev/detection_ctpn/

# copy output results
rsync -razv huasha@hb:ocr_dev/output/detection/* output/detection/