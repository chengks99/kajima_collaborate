#!/bin/bash
CUDA_VISIBLE_DEVICES=1 taskset -c 4,5,6,7 python3 kajima_camera2.py --redis-host 10.13.3.57 --redis-passwd ew@icSG23 --id 7115 --pcid 7000 -d -v
