#!/bin/bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,2,3 python3 -u kajima_camera2_ROI.py --redis-host 10.13.3.57 --redis-passwd ew@icSG23 --id 7108 --pcid 7000 -d -v
