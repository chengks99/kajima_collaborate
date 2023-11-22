#!/bin/bash
CUDA_VISIBLE_DEVICES=2 taskset -c 8,9,10,11 python3 kajima_camera2.py --redis-host 10.13.3.57 --redis-passwd ew@icSG23 --id 7003 --pcid 7000 -d -u -v
