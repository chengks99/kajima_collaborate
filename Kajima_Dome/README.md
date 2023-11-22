# Kajima-Dome
Dome camera module which:

* extract body features
* link body features with human ID
* report to server on body features

## SourceTree
|**directories**|**Descriptions**|
|:--:|---|
|**common**|common library files|
|**dome**|main dome camera file|
|**engine**|body feature extraction wrapper|
|**pose**|body posture detection|
|**rcnn**|RCNN network for reinforcement learning|
|**thermalcomfort**|thermal comfort module|
|**README.md**|this file|
|**requirements.txt**|require dependencies list file|

### Setup
Firstly install pip by:

```python
sudo apt install python3-pip
```

Install dependencies by:

```python
pip3 install -r requirements.txt
```

### Operation
Run below command to start backend server:

```python
python3 dome/kajima-dome.py --redis-host [CONTROL_PC_IP] --redis-passwd ewaic –d
```