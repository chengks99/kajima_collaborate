#!/bin/bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0,1,2,3 python3 kajima_camera2.py --redis-host 10.13.3.57 --redis-passwd ew@icSG23 --id 7114 --pcid 7000 -d -u
