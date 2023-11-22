# Kajima-server
Kajima server which:

* proxy between Redis and MySQL database
* handle result update
* handle plugin module

## SourceTree
|**directories**|**Descriptions**|
|:--:|---|
|**backend**|main server and plugin module|
|**common**|common library files|
|**config**|client configuration files|
|**mqtt**|MQTT module with Integrated Server(deprecated)|
|**aud_config.ini**|audio client list and basic configuration|
|**vis_config.ini**|Vision (camera/dome) client list and basic configuration|
|**config.ini**|system configuration include MySQL, MQTT setting|
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
python3 backend/backend-server.py --redis-passwd ewaic -d
```