import numpy as np 
from datetime import datetime
import torch
import inspect


class Configuration(object):

    def __init__(self):
        self.alias = None
        self.num_epochs = 10
        self.batch_size = 128
        self.optimizer = 'adam'
        self.use_cuda = True
        self.device_id = 0 
        self.early_stopping = 1
        self.loss = torch.nn.BCELoss
        self.debug = False
        self.sub_sample = 2.0/8 # proportion of training and test data to be used. Use None for full data.
        self.slack = True
        self.use_test = True if (self.debug is False) else False
        self.load_preproc_data = True # True - load preprocessed data from input folder. False - preprocess data and save as pickle into input folder.

    def __getitem__(cls, x):
        '''make configuration subscriptable'''
        return getattr(cls, x)

    def __setitem__(cls, x, v):
        '''make configuration subscriptable'''
        return setattr(cls, x, v)

    def get_attributes(self): 
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
 
        # store only not the default attribute __xx__
        attribute_tuple_list = [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]

        attribute_dict = {}
        for tup in attribute_tuple_list:
            key = tup[0]
            value = tup[1]
            if key == 'loss':
                value = str(value)
            # convert numpy value to float
            if type(value) == np.float64:
                value = float(value)
            attribute_dict[key] = value

        return attribute_dict

    def set_model_dir(self):
        now = datetime.now()

        time_info = f'{now.year}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}'
        self.model_dir = f'model_weights/{self.alias}-{time_info}.model'

    def attribute_to_integer(self):
        '''Convert the attributes in self.integer_attribute_list to integer'''

        for attribute in self.integer_attribute_list:
            self[attribute] = int(self[attribute])

    def set_config(self, config):
        for key in config:
            self[key] = config[key]
    
    def append_diff(self, config):
        ''' Add only new attributes'''
        config_keys = config.get_attributes().keys()
        self_keys = self.get_attributes().keys()
        for key in config_keys:
            if key not in self_keys:
                self[key] = config[key]


class NNConfiguration(Configuration):

    def __init__(self):
        super(NNConfiguration, self).__init__()
        self.categorical_emb_dim = 128

        self.alias = 'NN'
        self.optimizer = 'adam'
        self.learning_rate = 0.001
        self.weight_decay = 0
        self.sequence_length = 10
        self.sess_length = 30
        self.verbose = True
        self.hidden_dims = [256 , 128]
        self.dropout_rate = 0.1 # in transformer postional encoding and multihead attetion layers
        self.loss = torch.nn.BCELoss
        self.model = 3 # 0: GruNet1, 1: GruNet2, 2: TransformerNet1, 3: TransformerNet2
       

