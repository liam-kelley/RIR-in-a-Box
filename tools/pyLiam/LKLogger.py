import pandas as pd

class LKLogger():
    '''Liam Kelley's custom logger for csv files.
    Variables: filename, columns
    Methods:
    .check_if_line_in_log(dict_to_check)
    .add_line_to_log(dict_to_add)
    .get_df()
    '''
    def __init__(self, filename='./LKLogger.csv', columns_for_a_new_log_file=[]):
        # TODO rewrite this better
        self.filename=filename
        try:
            df=pd.read_csv(self.filename)
            print("found log file ", self.filename)
            if len(df.columns)==0 or len(df)==0:
                print("LKLogger : found log file is empty")
                raise Exception()
            self.columns=df.columns
        except:
            print("log file ", self.filename, " not found or empty.")
            if columns_for_a_new_log_file==[]:
                print("LKLogger : no columns specified for a new csv log file")
                raise Exception()
            else:
                self.columns=columns_for_a_new_log_file
                df=pd.DataFrame(columns=columns_for_a_new_log_file)
                df.to_csv(self.filename, index=False)
                print("initialized log file ", self.filename)

    def _check_if_keys_in_columns(self,checked_dict):
        '''check that all keys are in the columns'''
        for key in checked_dict.keys():
            if key not in self.columns:
                raise Exception("LKLogger : key ", key, " not in columns ", self.columns)

    def check_if_line_in_log(self, dict_to_check):
        '''
        Checks if a line is in the log already.
        dict_to_check is a dictionary with some of the same keys as the columns of the csv file.
        the values are the values we want to check if they are already in the csv file.
        For example we couldhave dict_to_check={'task': 'overfit', 'device': 'cpu', 'max_order': 5, 'time': 0.5, 'loss': 0.1}
        Returns true if there is that line in the csv file, false otherwise.
        '''
        self._check_if_keys_in_columns(dict_to_check)

        df=pd.read_csv(self.filename)
        filtered_dfs=[]
        for key in dict_to_check.keys():
            filtered_dfs.append(df[key]==dict_to_check[key])
        filtered_df=pd.concat(filtered_dfs, axis=1)
        full_true_lines=filtered_df.all(axis=1)
        if full_true_lines.any():
            return True
        else:
            return False
    
    def add_line_to_log(self, dict_to_add):
        '''
        Adds a line to the log, fills empty columns with None.
        dict_to_add is a dictionary with some of the same keys as the columns of the csv file.
        the values are the values we want to add to the csv file.
        They do not need to be within some 1-length lists.
        '''
        self._check_if_keys_in_columns(dict_to_add)
        
        for key in dict_to_add.keys(): #make them all lists if possible
            if type(dict_to_add[key])!=list:
                dict_to_add[key]=[dict_to_add[key]]

        # for every key not in dict_to_add but in self.columns, add a [None] to dict_to_add with the correct key
        for key in self.columns:
            if key not in dict_to_add.keys():
                dict_to_add[key]=[None]

        df=pd.read_csv(self.filename)
        df=pd.concat([df,pd.DataFrame(dict_to_add)], ignore_index=True)
        df.to_csv(self.filename, index=False)
        print("LKLogger: added log line to ",self.filename, end="\n\n")

    def get_df(self):
        return pd.read_csv(self.filename)

    def add_column_to_log(self):
        raise NotImplementedError()