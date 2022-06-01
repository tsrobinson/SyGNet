from src.sygnet.sygnet_requirements import *
from src.sygnet.sygnet_models import _MixedActivation
from src.sygnet.sygnet_dataloaders import *

import unittest
import random
import numpy as np

class TestSum(unittest.TestCase):

    def setUp(self):

        self.n_samples = 1000
        csv_input = pd.DataFrame(data = {
                'x1' : np.random.uniform(low = -2, high = -1, size=self.n_samples),
                'x2' : np.random.normal(loc = 100, scale = 200, size = self.n_samples),
                'x3' : np.random.uniform(low = 5, high = 10, size=self.n_samples),
                'y' : np.random.normal(loc = 0, scale = 10, size = self.n_samples),
                'pet' : random.choices(['cat','cat','dog','dog','bird'], k = self.n_samples), 
                'name' : random.choices(['bob','carl','sam','joe','mike'], k = self.n_samples),
                'bin' : random.choices([1,0], k = self.n_samples)
                })

        print(csv_input.head())

        training_data = GeneratedData(csv_input, conditional = False)

        class test_nn(nn.Module):
            def __init__(self, input_size, indices, funcs):
                super().__init__()
                # self.lin1 = nn.Linear(input_size, 200)
                # self.lin2 = nn.Linear(200,70)
                self.out_mix = _MixedActivation(indices, funcs, device='cpu')

            def forward(self, x):
                # x = self.lin1(x)
                # x = self.lin2(x)
                x = self.out_mix(x)
                return x

        test_model = test_nn(input_size = training_data.x.shape[1], indices = training_data.x_indxs, funcs = training_data.x_funcs)

        self.out_df = test_model(training_data.x)
        

    def test_softmax_group1(self):
        """
        Test that 3-category column constrained to 1 (check with 5 obs)
        """
        
        self.assertEqual(torch.sum(self.out_df[:,5:8]), self.n_samples)

    def test_softmax_group2(self):
        """
        Test that 5-category column constrained to 1 (check with 5 obs)
        """
        
        self.assertEqual(torch.sum(self.out_df[:,8:]), self.n_samples)

    def test_bin(self):
        """
        Test binary column (check with 5 obs)
        """

        # Binary is in col-position 4 since it occurs after categorical, so moved back to before cats
        
        bin_col = self.out_df.detach().numpy()[:,4]
        less_than_one = sum(bin_col <= 1)
        greater_than_zero = sum(bin_col >= 0)
        
        self.assertEqual(less_than_one + greater_than_zero, self.n_samples*2)

    def test_relu(self):
        """
        Test strictly-positive column (check with 5 obs)
        """
        
        relu_col = self.out_df.detach().numpy()[:,2]
        greater_than_zero = sum(relu_col >= 0)
        
        self.assertEqual(greater_than_zero, self.n_samples)

if __name__ == '__main__':
    unittest.main()
