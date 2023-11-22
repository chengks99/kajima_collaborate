#!/bin/bash
CUDA_VISIBLE_DEVICES=0 taskset -c 12,13,14,15 python3 kajima_dome.py --redis-host 10.13.3.57 --redis-passwd ew@icSG23 --id 7014 --pcid 7000 -d  
