from src.sygnet.sygnet_requirements import *
from src.sygnet.sygnet_models import _MixedActivation
from src.sygnet.sygnet_dataloaders import *

import unittest

class TestSum(unittest.TestCase):

    def setUp(self):
        csv_input = pd.read_csv('data/simulation_data_1.csv')

        csv_input = pd.concat([
            csv_input.iloc[0:5,:],
            pd.DataFrame(data = {'pet' : ['cat','cat','dog','dog','bird'], 'name' : ['bob','carl','sam','joe','mike'], 'bin' : [1,0,0,1,0]})
        ], axis=1)

        training_data = GeneratedData(csv_input, conditional = False, cond_cols = ['pet'])

        class test_nn(nn.Module):
            def __init__(self, input_size, indices, funcs):
                super().__init__()
                self.lin1 = nn.Linear(input_size, 200)
                self.lin2 = nn.Linear(200,70)
                self.out_mix = _MixedActivation(indices, funcs, device='cpu')

            def forward(self, x):
                x = self.lin1(x)
                x = self.lin2(x)
                x = self.out_mix(x)
                return x

        test_model = test_nn(input_size = training_data.x.shape[1], indices = training_data.x_indxs, funcs = training_data.x_funcs)

        self.out_df = test_model(training_data.x)
        

    def test_softmax_group1(self):
        """
        Test that 3-category column constrained to 1 (check with 5 obs)
        """
        
        self.assertEqual(torch.sum(self.out_df[:,5:8]), 5)

    def test_softmax_group2(self):
        """
        Test that 5-category column constrained to 1 (check with 5 obs)
        """
        
        self.assertEqual(torch.sum(self.out_df[:,8:]), 5)

    def test_bin(self):
        """
        Test binary column (check with 5 obs)
        """
        
        bin_col = self.out_df.detach().numpy()[:,4]
        less_than_one = sum(bin_col <= 1)
        greater_than_zero = sum(bin_col >= 0)
        
        self.assertEqual(less_than_one + greater_than_zero, 10)

if __name__ == '__main__':
    unittest.main()
