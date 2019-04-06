#!/bin/bash

PROJ_PATH=..
cd $PROJ_PATH

rsync -razv recognition-crnn/model/zh_model huasha@hb:ocr_dev/recognition-crnn/model/
rsync -razv detection-ctpn/checkpoints_mlt huasha@hb:ocr_dev/detection-ctpn/