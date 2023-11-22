#!/bin/bash
CUDA_VISIBLE_DEVICES=1 taskset -c 4,5,6,7 python3 -u kajima_camera2_ROI.py --redis-host 10.13.3.57 --redis-passwd ew@icSG23 --id 7109 --pcid 7000 -d -v
