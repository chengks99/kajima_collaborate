#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
miscellaneous utilities
'''
import platform

def get_my_ip ():
    ''' return the IP address used by the host '''
    import socket 
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except:
        return socket.gethostbyname(socket.gethostname())

def get_all_ip ():
    ''' return all th IP addresses associated with the host (does not work on Windows) '''
    from subprocess import check_output 
    if platform.system() == 'Linux':
        return check_output(['hostname', '-I']).decode().strip().split(' ')
    if platform.system() == 'Darwin':
        return ['192.168.1.23']

def get_best_match_ip (ipA, ipList):
    ''' return the address in ipList that is closest to ipA '''
    if len(ipList) <= 1:
        return ipList[0]
    if ':' in ipA: ipA = ipA.split(':',1)[0]
    cA = ipA.split('.')
    def match_score(cB):
        for i in range(4):
            if cA[i] != cB[i]: return i
        return 4
    b = (ipList[0], match_score(ipList[0].split('.')))
    for x in ipList[1:]:
        s = match_score(x.split('.'))
        if s > b[1]: b = (x, s)
    return b[0]

def copy_dict (d, fields, fieldnames={}):
    ''' return a copy of the dict %d with fields specified in %fields
        if %fieldnames is given, it should be a dict containing the 
        mapping of the field names in %d to a new field name
    '''
    fields = set([*fields, *fieldnames.keys()])
    return {
        fieldnames.get(f,f): d[f] for f in fields if f in d 
    }

if __name__ == '__main__':
    print(get_all_ip())
    # For testing
    A = '192.168.200.100'
    B = [ '192.168.1.20', '10.80.50.6', '192.168.200.56']
    print('{}: best match: {}'.format(A, get_best_match_ip(A,B)))
