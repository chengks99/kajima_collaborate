import sys
import os
import socket
import json
import logging
import ctypes
import pathlib
import platform
import multiprocessing as mp
import datetime as dt

import librosa
import torch
import torch.nn as nn
import numpy as np

from scipy.special import softmax

scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))

from adaptor import Adaptor
from jsonutils import json2str

sys.path.append(str(scriptpath.parent / 'microphone'))

from enhance_boost import Enhance_Boost
from vad import VAD
from asl import Audio_ASL
from sound_level import Sound_Level
from auemo.au_transform_infer import Audio_Transform
from auemo.models.vgg_m2 import VGG_M

def audio_socket_connection (port):
    try:
        HOST_IP = socket.gethostbyname(socket.gethostname())
        _tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _tcp.bind(("", port))
        logging.debug('Waiting for audio connection at port: {}'.format(port))
        _tcp.listen(1)
        conn, addr = _tcp.accept()
        logging.debug('Client {} connected to server {} at {}'.format(addr[0], HOST_IP, port))
        return conn, addr[0]
    except Exception as e:
        logging.error('Connection to server failed at port {}. {}'.format(port, e))
        exit(1)

class AudioControl(Adaptor):
    def __init__(self, args, **kw):
        self.args = args
        self._ipaddr = str(kw.get('address', ''))
        if self._ipaddr == '':
            logging.error('IP address cannot be empty string')
            exit(1)

        self.subscribe_channels = [
            'audio.{}.update'.format(self.args.id),
            'audio.{}.sqlquery'.format(self.args.id),
        ]
        self.micID = None
        Adaptor.__init__(self, args, **kw)
        self.load_config()
        self.start_listen_bus()
        self.run()
        self.pcid = self.args.pcid
        
        
        

    
    def load_config(self):
        logging.debug('Load configuration from Redis for {}'.format(self.args.id))
        cfg = Adaptor.get_redismsg_by_channel(self, '{}.config'.format(self.component_prefix))
        if cfg is None:
            logging.debug('No configuration for {} found in Redis, use config file {} instead'.format(self.args.id, self.args.config))
            if not os.path.isfile(self.args.config):
                logging.error('Unable to retrieve configuration file ...')
                exit(1)
            with open(self.args.config) as cfgf:
                self.cfg = json.load(cfgf)
        else:
            logging.debug('Configurtion in redis loaded ...')
            if 'pcid' in cfg: self.pcid = cfg['pcid']
            if 'config' in cfg:
                self.cfg = cfg['config']
            else:
                for key, val in cfg.items():
                    if key == 'id': continue
                    cfgPath = str(scriptpath.parent / val)
                    if not os.path.isfile(cfgPath):
                        logging.error('Unable to retrieve {} config from path: {}'.format(key, val))
                        exit(1)
                    with open(val) as cfgf:
                        self.cfg.update(json.load(cfgf))
        logging.debug('{} configuration: {}'.format(self.args.id, self.cfg))
    
    def run_init (self):

        self.init_params()
        self.init_module()

    def run (self):
        msg = Adaptor.get_redismsg_by_channel(self, '{}.detail-config'.format(self.component_prefix))
        if msg is None:
            logging.debug('Detail config not found, make request for detail config to backend server')
            # self.publish_redis_msg('{}.query'.format(self.component_prefix), {'msgType': 'init', 'address': self._ipaddr})
            self.micID = int(self.args.id.split("-")[-1])
            logging.debug(self.micID)
            self.run_init()


        else:
            logging.debug('Found detail config in redis bus')
            self.process_sql_result(msg)
            self.run_init()

    def init_params (self):
        try:
            self.fs = self.cfg['fs']
            self.window_asl = self.cfg['window_asl'] * self.fs
            self.window_auemo = int(1*self.fs)
            self.rec_dur = self.cfg['record_duration']
            self.vad_level = self.cfg['vad_level']
            self.num_src = self.cfg['num_source']
            self.no_classes = self.cfg['num_classes']
            self.channel = self.cfg['channel']

        except Exception as e:
            logging.error('Init params error: {}'.format(e))
            exit(1)
        
    def init_module (self):
        self.aud_enhance = Enhance_Boost(ref_noise=False, ref_noise_address=None)
        logging.debug('Enhance Boost module initialized ... ')
        self.vad_process = VAD(fs=self.fs, level=self.vad_level)
        logging.debug('VAD Process module initialized ... ')
        self.aud_localize = Audio_ASL(fs=self.fs)
        logging.debug('Audio ASL Process module initialized ... ')
   
    def process_redis_msg(self, ch, msg):
        if ch in self.subscribe_channels:
            if 'sqlquery' in ch:
                self.process_sql_result(msg)
                self.run_init()
    
    def process_sql_result (self, msg):
        if msg.get('type', '') == 'init':
            self.micID = msg.get('mic', {}).get('mic_id', None)

    def stream_audio (self, lock1, q, event, flag, conn):
        packet = 0
        while True:
            if flag.value == 1:
                data = conn.recv(int(self.fs * self.channel * 2), socket.MSG_WAITALL)
                if data == b"": break
                packet += 1
                lock1.acquire()
                q.put(data)
                event.value = 1
                lock1.release()
                logging.debug("Packet Numner: {}".format(packet))
    
    def auemo_infer (self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params = {
            'fs': self.fs,
            'time': self.window_auemo,
            'n_fft': 1024,
            'win_length': int(0.02 * self.fs),
            'hop_length': int(0.01 * self.fs),
            'targetloudness': -12,
            'platform_name': platform.system()
        }
        transform = Audio_Transform(device=self.device, para=params)
        audio_lv = Sound_Level(params)
        network = VGG_M(no_class=self.no_classes)

        device_id = [0] if torch.cuda.device_count() == 1 else [0, 1]
        network = nn.DataParallel(network, device_ids=device_id)
        network =network.to(self.device)
        modelf = scriptpath.parent / 'microphone/auemo/model_zoo/best_model.pt'
        if not modelf.is_file():
            logging.error('Unable to locate model file at {}'.format(str(modelf)))
        network = self.load_model(str(modelf), network)
        return transform, network, audio_lv

    def load_model(self, pre_trained_model_path, network):
        checkpoint = torch.load(pre_trained_model_path)
        # network.load_state_dict(checkpoint['model_state_dict'])
        network.load_state_dict(checkpoint['state_dict'])
        return network
        
    def raw_to_ip(self, data):
        dt = np.dtype(np.int16)
        dt = dt.newbyteorder('>')
        ip = np.frombuffer(data, dtype=dt) 
        ip = np.reshape(ip, (int(self.fs * self.rec_dur), self.channel))
        ip = np.array(ip, dtype=np.float32)
        ip = ip / (2 ** 15)
        return ip
    
    def data_preprocess(self, ip, buffer):
        update_size = int(self.fs * self.rec_dur)
        temp_buff = buffer
        temp_buff = np.vstack((temp_buff, ip))
        temp_buff = temp_buff[update_size:, :]
        buffer = temp_buff
        return buffer
    
    def normalize(self, ip):
        ip_norm = librosa.util.normalize(ip)
        return ip_norm
    
    def auemo_model_eval(self, model, input):
        model.eval()
        data = input 
        with torch.no_grad():
            data = data.to(self.device)
            outputs = model(data)
            predicted = outputs.to('cpu').detach().numpy()
            accuracy = softmax(predicted)*100
        return np.round(accuracy, 3)
    
    def emotion_variable(self,data1,data2,data3) :
        data1 = float(str(data1).split(".")[-1])
        data2 = float(str(data2).split(".")[-1])
        data3 = float(str(data3).split(".")[-1])


        if data1 >= data2 :
            cur_num = data1
            status = 0
        else :
            cur_num = data2
            status = 1

        if cur_num >= data3 :
            cur_num = cur_num 
            status = status
        else :
            cur_num = data3
            status = -1

        return status

    def audio_monitoring (self, lock1, q, event, flag):

        if self.no_classes == 7:
            classes = ['silence', 'clapping', 'laughing', 'scream-shout', 'conversation', 'happy', 'angry']
            new_classes = ['positive', 'neutral', 'negative']
        else:
            classes = ["positive", "negative", "neutral"]
            new_classes = classes
        
        transform, network, audio_lv = self.auemo_infer()
        logging.debug('Loaded audio monitoring model')

        buffer_asl = np.zeros((self.window_asl, self.channel))
        buffer_auemo = np.zeros((self.window_auemo, self.channel))

        flag.value = 1
        packet = 0
        while True:
            if event.value == 1:
                lock1.acquire()
                data = q.get()
                lock1.release()
                event.value = 0

                _ip = self.raw_to_ip(data)
                buffer_asl = self.data_preprocess(_ip, buffer_asl)
                buffer_auemo = self.data_preprocess(_ip, buffer_auemo)

                ip_norm = self.normalize(buffer_asl)
                enhance_ip1d = self.aud_enhance.enhance(ip_norm[:, 0])
                per_speech, vad_rec = self.vad_process.estimate(enhance_ip1d)
                audio_level = 0.0
                if per_speech > .6:
                    audio_level = audio_lv.main(ip_norm[:, 0])
                    x = transform.main(buffer_auemo[:, 0])
                    accuracy = self.auemo_model_eval(network, x)
                
                    auemo_output_csv = []
                    if self.no_classes == 7:
                        acc = accuracy[0]
                        n_acc = [acc[1] + acc[2] + acc[5], acc[0] + acc[4], acc[6] + acc[3]]
                        for na in n_acc:
                            auemo_output_csv.append('{:.2f}'.format(na))
                    else:
                        auemo_output_csv =['{:.2f}'.format(round(accuracy[0][i], 2)) for i in range(len(classes))]           
                else:
                    accuracy = [4., 1., 95.]
                    auemo_output_csv =['{:.2f}'.format(round(accuracy[i], 2)) for i in range(len(classes))]  
                
                est_posi = self.aud_localize.localise(ip_norm, self.num_src)
                packet += 1

                logging.debug('Estimated Angle: {} VAD: {}, Sound Level: {:.3f}'.format(est_posi, np.round(per_speech, 2), audio_level))

                status = self.emotion_variable(auemo_output_csv[0], auemo_output_csv[1], auemo_output_csv[2])
                if audio_level == 0 :
                    continue
                else :
                    _res = {
                        'status': status,
                        'audio_level': audio_level,
                        'est_posi': est_posi[0],
                        'mic_id': self.micID,
                        'timestamp': dt.datetime.now(),
                        'pcid': self.pcid
                    }
                    self.redis_conn.publish(
                        '{}.result'.format(self.component_prefix),
                        json2str(_res)
                    )
                    logging.debug('Publish result {}'.format(_res))
    
    def get_status (self):
        r = Adaptor.get_status(self)
        r['status'] = 'Normal'
        return r
    
    def get_info (self):
        r = Adaptor.get_info(self)
        return r

if __name__ == "__main__":
    import argsutils as au
    from adaptor import add_common_adaptor_args

    parser = au.init_parser('Kajima Audio Application')
    add_common_adaptor_args(
        parser,
        type='rpi-pi-mic',
        location='Office-2',
        id='audio-18',
        pcid=7000
    )
    args = au.parse_args(parser)

    DEBUG = False
    if DEBUG:
        audCtrl = AudioControl(args=args)
        exit(1)

    conn, addr = audio_socket_connection(args.sock)
    lock1 = mp.Lock()
    q = mp.Queue()
    event = mp.Value('i', 0)
    flag = mp.Value('i', 0)
    ip_addr_cString = mp.Value(ctypes.c_wchar_p, addr)

    audCtrl = AudioControl(args=args, address=ip_addr_cString.value)
    # print(audCtrl.no_classes)
    # exit()
    p_read = mp.Process(target=audCtrl.stream_audio, args=(lock1, q, event, flag, conn))
    p_process = mp.Process(target=audCtrl.audio_monitoring, args=(lock1, q, event, flag))

    p_read.start()
    p_process.start()

    p_read.join()
    p_process.join()