#!/bin/bash

# Create and activate the Python 3.8 conda environment (replace 'py38env' with your desired environment name)
conda activate py37

# Run the Python file (replace 'your_script.py' with your actual Python file)
python backend-server_utility_rate.py --redis-host 10.13.3.57 --redis-passwd ew@icSG23 -d
