import os
import numpy as np
import pandas as pd

import logging


class Database:
    def __init__(self, db_file = None,db_dict=None):
        if not db_file is None:
            self.db_file = db_file
            self.columns = ['name', 'features', 'person_details']

            # Read from file if it exists, else instantiate a new Database.
            if os.path.isfile(self.db_file):
                self.data = pd.read_pickle(self.db_file)
            else:
                self.data = pd.DataFrame(columns=self.columns)
        if not db_dict is None:
            self.data = pd.DataFrame(db_dict)

    def print_data(self):
        #print(self.data)
        logging.debug("{} identities in database.".format(len(self.data)))

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
        try :
            label = self.data['name'].iloc[index]
            details = self.data['person_details'].iloc[index]
        except :
            label = 'UnK'
            details = {'gender': np.nan, 'race': np.nan, 'age': np.nan}
        
        return label, details
