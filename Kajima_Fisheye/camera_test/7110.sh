#!/bin/bash
CUDA_VISIBLE_DEVICES=2 taskset -c 12,13,14,15 python3 -u kajima_camera2_ROI.py --redis-host 10.13.3.57 --redis-passwd ew@icSG23 --id 7110 --pcid 7000 -d
