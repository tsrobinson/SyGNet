import unittest
from src.sygnet.sygnet_interface import *

class TestSum(unittest.TestCase):

    def setUp(self):
        self.input_data = pd.read_csv("data/simulation_data_1.csv",delimiter=",").iloc[0:100,:]

    def test_basic(self):
        """
        Test without conditional labels or Wasserstein loss
        """

        model = SygnetModel(mode="basic")
        model.fit(self.input_data, epochs=1, batch_size=self.input_data.shape[0])
        self.assertEqual(model.sample(5).shape, (5,4))

    def test_wgan(self):
        """
        Test without conditional labels
        """

        model = SygnetModel(mode="wgan")
        model.fit(self.input_data, epochs=1, batch_size=self.input_data.shape[0])
        self.assertEqual(model.sample(5).shape, (5,4))
    
    def test_cgan(self):
        """
        Test with conditional labels
        """

        model = SygnetModel(mode="cgan")
        model.fit(self.input_data, epochs=1, batch_size=self.input_data.shape[0], cond_cols =['x3'])
        self.assertEqual(model.sample(5, labels = self.input_data.loc[:4,'x3']).shape, (5,4))