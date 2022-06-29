from src.sygnet.sygnet_requirements import *
from src.sygnet.sygnet_tune import *
import random

import unittest

class TestSum(unittest.TestCase):

    def setUp(self):
        input_data = pd.read_csv("data/simulation_data_1.csv",delimiter=",").iloc[0:100,:]
        input_data['cat'] = random.choices(['a','b','c'],k=input_data.shape[0])
        input_data['cat_lat1'] = random.choices(['x','y','z'],k=input_data.shape[0])
        input_data['cat_lat2'] = random.choices(['q','r'],k=input_data.shape[0])
        input_data['bin1'] = random.choices([1,0], k=input_data.shape[0])

        self.input_data = input_data

    def test_tuner(self):
        """
        Test tuner
        """
        
        tuning_results = tune(
            parameter_dict = {'hidden_nodes':[[32,32],[32]], 'dropout_p':[0.1,0.2]},
            data = self.input_data,
            test_func = critic_loss, 
            runs = 2,
            mode = "wgan",
            k = 2,
            tuner = "random",
            n = 100,
            epochs = 1,
            seed = 89,
            device = 'cpu')

        self.assertEqual(tuning_results.shape[0], 4)
        