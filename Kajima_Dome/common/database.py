import os
import numpy as np
import pandas as pd
# import datetime
import time

class FaceDatabase:
    def __init__(self, db_file=None, db_dict=None):
        if not db_file is None:
            self.db_file = db_file
            self.columns = ['name', 'features']

            # Read from file if it exists, else instantiate a new Database.
            if os.path.isfile(self.db_file):
                self.data = pd.read_pickle(self.db_file)
            else:
                self.data = pd.DataFrame(columns=self.columns)
        else:
            self.data = pd.DataFrame(db_dict)
            self.columns = ['features', 'eID']
    def print_data(self):
        print(self.data)
        print("{} identities in database.".format(len(self.data)))

    def is_empty(self):
        return self.data.empty

    def is_name_in_database(self, name):
        if name in self.data['name'].values:
            return True
        return False

    def add_data(self, name, feat):
        # Update feature vector if the name exists.
        # If it doesn't exist, append {name, feature} as a new row.
        # Note that this changes the data in runtime memory, not the external .pkl file.
        if self.is_name_in_database(name):
            print("Overwriting feature vector as the ID already exists: {}".format(name))
            self.data.loc[(self.data['name'] == name), 'features'] = [feat]
            return False
        else:
            row = {'name': name, 'features': feat}
            self.data = self.data.append(row, ignore_index=True)
            return True

    def rename_data(self, old_name, new_name):
        # Updates name if it exists in the database.
        if self.is_name_in_database(old_name):
            if not self.is_name_in_database(new_name):
                self.data.loc[(self.data['name'] == old_name), 'name'] = new_name
                return 0
            else:
                print("Error: {} already exists in the database.".format(new_name))
                return 1
        else:
            print("Error: {} does not exist in the database.".format(old_name))
            return 2

    def remove_data(self, name):
        # Updates the database to one without the name.
        # Returns the removed row if it exists.
        # If it doesn't exist, no update and return None.
        if self.is_name_in_database(name):
            ret = self.data[self.data['name'] == name]
            self.data = self.data[self.data['name'] != name]
            print("Removed {} from database.".format(name))
            return ret
        else:
            print("Error: {} does not exist in the database.".format(name))
            return None

    def save_database(self, save_file=None):
        # Commit the current self.data to a .pkl file.
        if save_file:
            path = save_file
        else:
            path = self.db_file

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        self.data.to_pickle(path)
        string = "Database saved to {}. {} identities in database.".format(path, len(self.data))
        print(string)
        return string

    def get_features_as_array(self):
        # Returns all feature vectors as a (N, 1, DIM) numpy array.
        if self.data.empty: return None
        return np.stack(self.get_features_as_list())

    def get_features_as_list(self):
        # Returns all feature vectors as a list of numpy arrays of length N.
        if self.data.empty: return None
        return self.data['features'].tolist()

    def get_name_list(self):
        return self.data['name'].tolist()

    def get_name_from_index(self, index):
        # Retrieve the name string without worrying about pandas functions.
        return self.data['name'].iloc[index]

class FaceDatabasePersonal:
    def __init__(self, db_file=None, db_dict=None):
        if not db_file is None:
            self.db_file = db_file
            self.columns = ['name', 'features', 'gender', 'race', 'age'] 

            # Read from file if it exists, else instantiate a new Database.
            if os.path.isfile(self.db_file):
                self.data = pd.read_pickle(self.db_file)
            else:
                self.data = pd.DataFrame(columns=self.columns)
        else:
            self.data = pd.DataFrame(db_dict)
            self.columns = ['features', 'name', 'gender', 'race', 'age']
        self.print_data()
    def print_data(self):
        print(self.data)
        print("{} identities in database.".format(len(self.data)))

    def is_empty(self):
        return self.data.empty

    def is_name_in_database(self, name):
        if name in self.data['name'].values:
            return True
        return False

    def add_data(self, name, feat):
        # Update feature vector if the name exists.
        # If it doesn't exist, append {name, feature} as a new row.
        # Note that this changes the data in runtime memory, not the external .pkl file.
        if self.is_name_in_database(name):
            print("Overwriting feature vector as the ID already exists: {}".format(name))
            self.data.loc[(self.data['name'] == name), 'features'] = [feat]
            return False
        else:
            row = {'name': name, 'features': feat}
            self.data = self.data.append(row, ignore_index=True)
            return True

    def rename_data(self, old_name, new_name):
        # Updates name if it exists in the database.
        if self.is_name_in_database(old_name):
            if not self.is_name_in_database(new_name):
                self.data.loc[(self.data['name'] == old_name), 'name'] = new_name
                return 0
            else:
                print("Error: {} already exists in the database.".format(new_name))
                return 1
        else:
            print("Error: {} does not exist in the database.".format(old_name))
            return 2

    def remove_data(self, name):
        # Updates the database to one without the name.
        # Returns the removed row if it exists.
        # If it doesn't exist, no update and return None.
        if self.is_name_in_database(name):
            ret = self.data[self.data['name'] == name]
            self.data = self.data[self.data['name'] != name]
            print("Removed {} from database.".format(name))
            return ret
        else:
            print("Error: {} does not exist in the database.".format(name))
            return None

    def save_database(self, save_file=None):
        # Commit the current self.data to a .pkl file.
        if save_file:
            path = save_file
        else:
            path = self.db_file

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        self.data.to_pickle(path)
        string = "Database saved to {}. {} identities in database.".format(path, len(self.data))
        print(string)
        return string

    def get_features_as_array(self):
        # Returns all feature vectors as a (N, 1, DIM) numpy array.
        if self.data.empty: return None
        return np.stack(self.get_features_as_list())

    def get_features_as_list(self):
        # Returns all feature vectors as a list of numpy arrays of length N.
        if self.data.empty: return None
        return self.data['features'].tolist()

    def get_name_list(self):
        return self.data['name'].tolist()

    def get_name_from_index(self, index):
        # Retrieve the name string without worrying about pandas functions.
        return self.data['name'].iloc[index]

    def get_details_from_index(self, index):
        # Retrieve the name string without worrying about pandas functions.
        details = {'gender': self.data['gender'].iloc[index], 
                   'race': self.data['race'].iloc[index], 
                   'age': self.data['age'].iloc[index]}
        
        return self.data['name'].iloc[index], details

class BodyDatabaseFront:
    def __init__(self, db_file=None, db_dict=None, body_feat_life=32400):
        self.update_flag = False
        self.body_feat_life = body_feat_life
        if not db_file is None:
            self.db_file = db_file
            
            # self.int_cols = ['name', 'featuresf', 'featuresb', 'timeStamp']
            self.ext_cols = ['name', 'features', 'timeStamp', 'person_details']

            # Read from file if it exists, else instantiate a new Database.
            if os.path.isfile(self.db_file):
                #reading existing database
                print("Reading existing database ", self.db_file)
                self.data = pd.read_pickle(self.db_file)
            else:
                self.data = pd.DataFrame(columns=self.ext_cols)
                print(self.data)
        if not db_dict is None:
            self.data = pd.DataFrame(db_dict)

            # x2=datetime.datetime.now()
            # x2.hour, x2.minute
            # ts2 = x2.timestamp()
            # readable = datetime.datetime.fromtimestamp(ts2).isoformat()
            # readable.hour[]
            # d = datetime.datetime.fromisoformat(s)
            # readable = datetime.datetime.fromtimestamp(ts)

    def print_data(self):
        print(self.data)
        print("{} identities in database.".format(len(self.data)))

    def is_empty(self):
        return self.data.empty

    def is_name_in_database(self, name):
        if name in self.data['name'].values:
            return True
        return False

    def add_data(self, name, featf, person_details):
        # Update feature vector if the name exists.
        # If it doesn't exist, append {name, feature} as a new row.
        # Note that this changes the data in runtime memory, not the external .pkl file.

        # timenow = datetime.datetime.now()
        # timestamp = timenow.timestamp()
        timestamp = time.time() #time in seconds

        #add multiple body features instead of replacment as discussed in Nov 2021

        # if self.is_name_in_database(name):
        #     idx = self.data.index[self.data['name'] == name].item()
        #     print("Overwriting feature vector as the ID already exists: {}".format(name))
        #     self.data.at[idx, 'featuresf'] = featf 
        #     self.data.at[idx, 'featuresb'] = featb 
        #     self.data.at[idx, 'timeStamp'] = timestamp 
        #     return False
        # else:
        #     row = {'name': name, 'featuresf': featf, 'featuresb': featb, 'timeStamp': timestamp}
        #     # row = {'name': name, 'featuresf': featf, 'featuresb': featb}
        #     self.data = self.data.append(row, ignore_index=True)
        #     return True

        row = {'name': name, 'features': featf, 'timeStamp': timestamp, 'person_details':person_details}
        self.data = self.data.append(row, ignore_index=True)
        self.update_flag = True

        return True

    def rename_data(self, old_name, new_name):
        # Updates name if it exists in the database.
        if self.is_name_in_database(old_name):
            if not self.is_name_in_database(new_name):
                self.data.loc[(self.data['name'] == old_name), 'name'] = new_name
                return 0
            else:
                print("Error: {} already exists in the database.".format(new_name))
                return 1
        else:
            print("Error: {} does not exist in the database.".format(old_name))
            return 2

    def remove_data(self, name):
        # Updates the database to one without the name.
        # Returns the removed row if it exists.
        # If it doesn't exist, no update and return None.
        if self.is_name_in_database(name):
            ret = self.data[self.data['name'] == name]
            self.data = self.data[self.data['name'] != name]
            print("Removed {} from database.".format(name))
            return ret
        else:
            print("Error: {} does not exist in the database.".format(name))
            return None

    def _remove_stale_data(self):
        #keep all features less than body_feat_life (9hours=32400seconds)
        curr_time = time.time()
        self.data = self.data[(curr_time - self.data['timeStamp']) < self.body_feat_life]
        print("Removed stale data from database.")

    def save_database(self, save_file=None):
        # Commit the current self.data to a .pkl file.
        if self.update_flag:
            if save_file:
                path = save_file
            else:
                path = self.db_file

            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            print("LEN of self data in database ", len(self.data))

            self.update_flag = False
            self.data.to_pickle(path)
            string = "Database saved to {}. {} identities in database.".format(path, len(self.data))
            print(string)

    def get_data_frame_as_array(self, col):
        # Returns all feature vectors as a (N, 1, DIM) numpy array.
        if self.data.empty: return None
        return np.stack(self.get_data_frame_as_list(col))

    def get_data_frame_as_list(self, col):
        return self.data[col].tolist()

    def get_name_from_index(self, index):
        # Retrieve the name string without worrying about pandas functions.
        return self.data['name'].iloc[index]

class BodyDatabaseFrontBack:
    def __init__(self, db_file, body_feat_life):
        self.db_file = db_file
        self.body_feat_life = body_feat_life
        self.int_cols = ['name', 'featuresf', 'featuresb', 'timeStamp']
        self.ext_cols = ['name', 'features', 'timeStamp']

        # Read from file if it exists, else instantiate a new Database.
        if os.path.isfile(self.db_file):
            #reading existing database
            print("Reading existing database ", self.db_file)
            self.data = self.__read_convert_database(self.db_file)
        else:
            self.data = pd.DataFrame(columns=self.int_cols)
            print(self.data)

            # x2=datetime.datetime.now()
            # x2.hour, x2.minute
            # ts2 = x2.timestamp()
            # readable = datetime.datetime.fromtimestamp(ts2).isoformat()
            # readable.hour[]
            # d = datetime.datetime.fromisoformat(s)
            # readable = datetime.datetime.fromtimestamp(ts)

    def print_data(self):
        print(self.data)
        print("{} identities in database.".format(len(self.data)))

    def is_empty(self):
        return self.data.empty

    def is_name_in_database(self, name):
        if name in self.data['name'].values:
            return True
        return False

    def add_data(self, name, featf, featb):
        # Update feature vector if the name exists.
        # If it doesn't exist, append {name, feature} as a new row.
        # Note that this changes the data in runtime memory, not the external .pkl file.

        # timenow = datetime.datetime.now()
        # timestamp = timenow.timestamp()
        timestamp = time.time() #time in seconds

        #add multiple body features instead of replacment as discussed in Nov 2021

        # if self.is_name_in_database(name):
        #     idx = self.data.index[self.data['name'] == name].item()
        #     print("Overwriting feature vector as the ID already exists: {}".format(name))
        #     self.data.at[idx, 'featuresf'] = featf 
        #     self.data.at[idx, 'featuresb'] = featb 
        #     self.data.at[idx, 'timeStamp'] = timestamp 
        #     return False
        # else:
        #     row = {'name': name, 'featuresf': featf, 'featuresb': featb, 'timeStamp': timestamp}
        #     # row = {'name': name, 'featuresf': featf, 'featuresb': featb}
        #     self.data = self.data.append(row, ignore_index=True)
        #     return True

        row = {'name': name, 'featuresf': featf, 'featuresb': featb, 'timeStamp': timestamp}
        # row = {'name': name, 'featuresf': featf, 'featuresb': featb}
        self.data = self.data.append(row, ignore_index=True)
        return True

    def rename_data(self, old_name, new_name):
        # Updates name if it exists in the database.
        if self.is_name_in_database(old_name):
            if not self.is_name_in_database(new_name):
                self.data.loc[(self.data['name'] == old_name), 'name'] = new_name
                return 0
            else:
                print("Error: {} already exists in the database.".format(new_name))
                return 1
        else:
            print("Error: {} does not exist in the database.".format(old_name))
            return 2

    def remove_data(self, name):
        # Updates the database to one without the name.
        # Returns the removed row if it exists.
        # If it doesn't exist, no update and return None.
        if self.is_name_in_database(name):
            ret = self.data[self.data['name'] == name]
            self.data = self.data[self.data['name'] != name]
            print("Removed {} from database.".format(name))
            return ret
        else:
            print("Error: {} does not exist in the database.".format(name))
            return None

    def _remove_stale_data(self):
        #keep all features less than body_feat_life (9hours=32400seconds)
        curr_time = time.time()
        self.data = self.data[(curr_time - self.data['timeStamp']) < self.body_feat_life]
        print("Removed stale data from database.")

    def convert_save_database(self, save_file=None):
        # remove stale data before saving
        self._remove_stale_data()

        # self.ext_cols = ['name', 'features']
        names = self.data['name'].tolist()
        featf = self.data['featuresf'].tolist()
        featb = self.data['featuresb'].tolist()
        timestamp = self.data['timeStamp'].tolist()

        data_ext = pd.DataFrame(columns=self.ext_cols)
        print(data_ext, self.ext_cols)
        for i in range(len(names)):
            row = {'name': names[i], 'features': featf[i], 'timeStamp': timestamp[i]}
            data_ext = data_ext.append(row, ignore_index=True)
            row = {'name': names[i], 'features': featb[i], 'timeStamp': timestamp[i]}
            data_ext = data_ext.append(row, ignore_index=True)

        # Commit the current self.data to a .pkl file.
        if save_file:
            path = save_file
        else:
            path = self.db_file

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        data_ext.to_pickle(path)
        # string = "Database saved to {}. {} identities in database.".format(path, len(self.data))
        print("Database saved to {}. {} identities in database.".format(path, len(data_ext)))
        # return string

    def __read_convert_database(self, read_file=None):
        # read the current .pkl file to self.data.
        ext_data = pd.read_pickle(self.db_file)
        print("Ext data ", ext_data)

        names = ext_data['name'].tolist()
        features = ext_data['features'].tolist()
        timestamp = ext_data['timeStamp'].tolist()

        #as same name for backside and frontside, remove duplicate names (get unique name list)
        new_names = []
        for x in names:
            if x not in new_names:
                new_names.append(x)
        # new_names = [new_names.append(x) for x in names if x not in new_names]

        int_data = pd.DataFrame(columns=self.int_cols)
        for i in range(len(new_names)):
            row = {'name': new_names[i], 'featuresf': features[2*i], 'featuresb': features[2*i+1], 'timeStamp': timestamp[2*i]}
            int_data = int_data.append(row, ignore_index=True)

        return int_data

    def get_data_frame_as_array(self, col):
        # Returns all feature vectors as a (N, 1, DIM) numpy array.
        if self.data.empty: return None
        return np.stack(self.get_data_frame_as_list(col))

    def get_data_frame_as_list(self, col):
        return self.data[col].tolist()

    def get_name_from_index(self, index):
        # Retrieve the name string without worrying about pandas functions.
        return self.data['name'].iloc[index]
