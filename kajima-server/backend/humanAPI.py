import requests
import logging
import json

class HumanAPI(object):
    def __init__(self, dets):
        logging.debug('Init Human HTTP API module')
        self.url = 'https://{}:{}/api'.format(dets.get('ip'), dets.get('port'))
        self.header = self.login(dets.get('username'), dets.get('password'))
        logging.debug(self.header)
        if self.header is None:
            logging.error('Failed to retrieve token')
            exit(1)
    
    def req_http_post (self, url, data, header=None, verify=False):
        try:
            if header is None:
                r = requests.post(url, data=data, verify=verify)
            else:
                r = requests.post(url, headers=header, data=data, verify=verify)
            r.raise_for_status()
            return (json.loads(r.text))
        except requests.exceptions.HTTPError as e:
            logging.error(e.response.text)
            return None
    
    def req_http_get (self, url, data, header=None, verify=False):
        try:
            if header is None:
                r = requests.get(url, params=data, verify=verify)
            else:
                r = requests.get(url, headers=header, params=data, verify=verify)
            r.raise_for_status()
            return (json.loads(r.text))
        except requests.exceptions.HTTPError as e:
            logging.error(e.response.text)
            return None

    def login (self, username, password):
        r = self.req_http_post(
            self.url + '/login', 
            {'username': username, 'password': password, 'isKeepLogin': True}
        )
        if not r is None:
            if not 'token' in r: return None
            return {'x-access-token': r['token']}
        else:
            raise ValueError('Unable to login')
    
    def get_human_data (self, pageSize=10, pageNum=1):
        r = self.req_http_get(
            self.url + '/data/get-human-data',
            {'pageSize': pageSize, 'pageNum': pageNum},
            self.header,
        )
        if not r is None:
            logging.debug(r)
        else:
            raise ValueError('Unable to retrieve human data')
    
    def add_human_data (self, data={}):
        if not data:
            logging.error('empty dict when calling add_human_data API')
        else:
            r = self.req_http_post(
                self.url + '/data/add-human-data',
                data=data,
                header=self.header,
            )
            if not r is None:
                logging.debug(r)
            else:
                raise ValueError('Unable to add human data')

    def update_human_data (self, data={}):
        if not data:
            logging.error('empty dict when calling update_human_data API')
        else:
            r = self.req_http_post(
                self.url + '/data/update-human-data',
                data=data,
                header=self.header,
            )
            if not r is None:
                logging.debug(r)
            else:
                raise ValueError('Unable to update human data')

if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG)
    dets = {
        'ip': '10.13.3.250',
        'port': 8443,
        'username': 'systemAdmin1',
        'password': 'Prvcd@2023',
    }

    _api = HTTPAPI(dets)
    _api.get_human_data()