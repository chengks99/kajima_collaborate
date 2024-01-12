#!/bin/bash
CUDA_VISIBLE_DEVICES=2 taskset -c 8,9,10,11 python3 -u kajima_camera2_ROI.py --redis-host 10.13.3.57 --redis-passwd ew@icSG23 --id 7107 --pcid 7000 -d -v
