#!/bin/bash
CUDA_VISIBLE_DEVICES=1 taskset -c 16,17,18,19 python3 kajima_dome.py --redis-host 10.13.3.57 --redis-passwd ew@icSG23 --id 7015 --pcid 7000 -d
