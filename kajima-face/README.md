# Kajima-Face
Face feature extraction module which:

* read face image from files
* extract features
* send to server for database insertion

## SourceTree
|**directories**|**Descriptions**|
|:--:|---|
|**common**|common library files|
|**engine**|face feature extraction wrapper|
|**feature**|main face feature extraction file|
|**rcnn**|RCNN network for reinforce learning|
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
python3 kajima/kajima-face.py --redis-host [CONTROL_PC_IP] --redis-passwd ewaic –d
```