import mysql.connector
import logging

class SQLDatabase(object):
    def __init__(self, args) -> None:
        self.args = args
        self.db, self.cur = None, None
        self.start_connection(args)
    
    def start_connection (self, args):
        try:
            if not self.db is None:
                self.db.close()
                self.cur.close()
            self.db = mysql.connector.connect(
                user=args.dbuser,
                password=args.dbpwd,
                host=args.dbhost,
                database=args.dbtbl,
                connection_timeout=1000,
            )
            self.cur = self.db.cursor()
            logging.debug('SQL connected to {}'.format(args.dbhost))
        except Exception as e:
            logging.error('SQL Connection Failed...')
            logging.error(e)
    
    def connection (self): 
        if self.db is None: return False
        #return False if self.db is None else True
        #logging.debug('Checking SQL connection')
        res = self.db.is_connected()
        #logging.debug('SQL connection {}: {}'.format(self.db, res))
        return res

    @property
    def cursor (self):
        return self.cur
    
    def commit (self):
        if not self.connection():
            self.start_connection(self.args)
        self.db.commit()
    
    def close (self, commit=True):
        if commit: self.commit()
        self.db.close()
    
    def execute (self, msg, data=None, commit=False):
        #logging.debug('Execute Database: {}. Commit={}'.format(msg.split(' ')[0], commit))
        if not self.connection():
            logging.error('SQL not connected, Reconnecting')
            self.start_connection(self.args)
        #print ('-------------- connection OK')
        try:
            cur = self.db.cursor()
            if data is None:
                self.cur.execute(msg)
            else:
                #print ('-------------- start execute')
                self.cur.execute(msg, data)
            self.error = False
        except Exception as e:
            logging.error(e)
            self.error = True
        if commit: self.commit()
        #print ('############################')
    
    def fetchall (self):
        return self.cur.fetchall()
    
    def query (self, msg, data=None):
        self.execute(msg, data)
        return self.cur.fetchall() if not self.error else None

if __name__ == "__main__":
    from argparse import Namespace
    mydict = {
        'dbuser': 'root',
        'dbpwd': 'Welcome123',
        'dbhost': 'localhost',
        'dbtbl': 'mockup_db',
    }
    ns = Namespace(**mydict)

    sql = SQLDatabase(ns)
    cur = sql.query('SELECT * FROM cam_table_demo WHERE cam_id = 18')
    print (cur)