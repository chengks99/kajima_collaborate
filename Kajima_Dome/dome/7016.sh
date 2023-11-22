#!/bin/bash
CUDA_VISIBLE_DEVICES=2 taskset -c 20,21,22,23 python3 kajima_dome.py --redis-host 10.13.3.57 --redis-passwd ew@icSG23 --id 7016 --pcid 7000 -d -v
