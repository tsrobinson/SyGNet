from src.sygnet.sygnet_requirements import *
from src.sygnet.sygnet_interface import *
from src.sygnet.sygnet_loader import *
import random

import unittest
import tempfile
import os 

class TestSum(unittest.TestCase):

    def setUp(self):
        input_data = pd.read_csv("data/simulation_data_1.csv",delimiter=",").iloc[0:100,:]
        input_data['cat'] = random.choices(['a','b','c'],k=input_data.shape[0])
        input_data['cat_lat1'] = random.choices(['x','y','z'],k=input_data.shape[0])
        input_data['cat_lat2'] = random.choices(['q','r'],k=input_data.shape[0])
        input_data['bin1'] = random.choices([1,0], k=input_data.shape[0])

        self.input_data = input_data

    def test_save_and_load(self):
        """
        Test saving and loading functionality
        """
        
        model0 = SygnetModel(mode = "wgan", hidden_nodes=[8,8])
        with tempfile.TemporaryDirectory() as tmpdirname:
            model0.fit(self.input_data, epochs = 1, save_model=True, save_loc = r"{}".format(tmpdirname))
            new_file = os.listdir(tmpdirname)[0]
            model1 = load(Path(tmpdirname) / new_file)

        self.assertTrue(
            torch.equal(
                model0.generator.state_dict()['linears.1.weight'],
                model1.generator.state_dict()['linears.1.weight']
            )
        )
        