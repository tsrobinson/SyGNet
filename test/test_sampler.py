from src.sygnet.sygnet_requirements import *
from src.sygnet.sygnet_interface import *
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

    def test_wgan(self):
        """
        Test without conditional labels
        """
        test_mod = SygnetModel(mode = "wgan")
        test_mod.fit(self.input_data, batch_size = self.input_data.shape[0], epochs = 1)

        with self.subTest():
            self.assertEqual(
                test_mod.sample(5, decode = True, as_pandas = True).columns.tolist(),
                ['x1','x2','x3','y','bin1','cat','cat_lat1','cat_lat2']
            )

        with self.subTest():
            uniq_vals = test_mod.sample(5, decode = True, as_pandas = True)['cat'].unique()
            self.assertTrue(set(uniq_vals).issubset(set(['a','b','c'])))

        with self.subTest():
            self.assertEqual(
                test_mod.sample(1, decode = True, as_pandas = False).shape[1],
                8
            )
        with self.subTest():
            self.assertEqual(
                test_mod.sample(5, decode = False, as_pandas = True).columns.tolist(),
                ['x1','x2','x3','y','bin1','cat_a','cat_b','cat_c','cat_lat1_x','cat_lat1_y','cat_lat1_z','cat_lat2_q','cat_lat2_r']
            )

        with self.subTest():
            self.assertEqual(
                test_mod.sample(1, decode = False, as_pandas = False).shape[1],
                13
            )

    def test_conditional_no_labels(self):
        """
        Test conditional generator without provided labels
        """

        test_mod = SygnetModel(mode = "cgan")
        test_mod.fit(self.input_data, cond_cols = ["cat"], batch_size = self.input_data.shape[0], epochs = 1)
        
        with self.assertRaises(ValueError):
            test_mod.sample(5, decode = True, as_pandas = True)

    def test_conditional_many_labels(self):
        """
        Test conditional generator with too many labels
        """

        test_mod = SygnetModel(mode = "cgan")
        test_mod.fit(self.input_data, cond_cols = ["cat"], batch_size = self.input_data.shape[0], epochs = 1)
        
        with self.assertRaises(RuntimeError):
            test_mod.sample(3, decode = True, as_pandas = True, labels = self.input_data.loc[0:4,'cat'])

    def test_conditional_few_labels(self):
        """
        Test conditional generator with not enough labels
        """

        test_mod = SygnetModel(mode = "cgan")
        test_mod.fit(self.input_data, cond_cols = ["cat"], batch_size = self.input_data.shape[0], epochs = 1)
        
        with self.assertRaises(RuntimeError):
            test_mod.sample(5, decode = True, as_pandas = True, labels = self.input_data.loc[0:2,'cat'])

    def test_conditional_badly_ordered(self):
        """
        Test conditional generator with badly ordered columns in sampler
        """

        test_mod = SygnetModel(mode = "cgan")
        test_mod.fit(self.input_data, cond_cols = ["cat","bin1",'x3'], batch_size = self.input_data.shape[0], epochs = 1)
        sample_labels = self.input_data.loc[0:4, ['bin1','cat','x3']]

        with self.subTest(msg="Check column names (decoded | pandas)"):
            self.assertEqual(
                test_mod.sample(5, decode = True, as_pandas = True, labels = sample_labels).columns.tolist(),
                ['x1','x2','y','cat_lat1','cat_lat2','bin1','cat','x3']
            )
        
        with self.subTest(msg="Check column values (decoded | pandas)"):
            uniq_vals = test_mod.sample(5, decode = True, as_pandas = True, labels = sample_labels)['cat'].unique()
            self.assertTrue(set(uniq_vals).issubset(set(['a','b','c'])))

        with self.subTest(msg="Check column shape (decoded | numpy)"):
            self.assertEqual(
                test_mod.sample(5, decode = True, as_pandas = False, labels = sample_labels).shape[1],
                8
            )
        with self.subTest(msg="Check column names (encoded | pandas)"):
            self.assertEqual(
                test_mod.sample(5, decode = False, as_pandas = True, labels = sample_labels).columns.tolist(),
                ['x1','x2','y','cat_lat1_x','cat_lat1_y','cat_lat1_z','cat_lat2_q','cat_lat2_r','bin1','x3','cat_a','cat_b','cat_c',]
            )

        with self.subTest(msg="Check column shape (encoded | numpy)"):
            self.assertEqual(
                test_mod.sample(5, decode = False, as_pandas = False, labels = sample_labels).shape[1],
                13
            )

    def test_cond_no_cat(self):
        """
        Test conditional generator with only numeric conditional labels
        """

        test_mod = SygnetModel(mode = "cgan")
        test_mod.fit(self.input_data, cond_cols = ["x1",'x2','x3'], batch_size = self.input_data.shape[0], epochs = 1)
        sample_labels = self.input_data.loc[0:4, ['x1','x2','x3']]

        out_dec_pd = test_mod.sample(5, decode = True, as_pandas = True, labels = sample_labels)
        out_enc_np = test_mod.sample(5, decode = False, as_pandas = False, labels = sample_labels)

        with self.subTest(msg="Check column names (decoded | pandas)"):
            self.assertEqual(
                out_dec_pd.columns.tolist(),
                ['y','bin1','cat','cat_lat1','cat_lat2','x1','x2','x3']
            )

        with self.subTest(msg="Check softmax (encoded | numpy)"):
            cat_col = out_enc_np[:,2:5]
            self.assertEqual([round(num,5) for num in cat_col.sum(axis=1).tolist()], [1.00000 for i in range(5)])


    def test_data_mix_order(self):
        """
        Test conditional generator with badly ordered columns in sampler
        """

        input_data = self.input_data[['x3','cat_lat1', 'x2', 'bin1', 'y','cat_lat2','x1', 'cat']]

        test_mod = SygnetModel(mode = "cgan")
        test_mod.fit(input_data, cond_cols = ["cat",'x3'], batch_size = input_data.shape[0], epochs = 1)
        sample_labels = input_data.loc[0:4, ['cat','x3']]
        out_dec_pd = test_mod.sample(5, decode = True, as_pandas = True, labels = sample_labels)
        out_enc_pd = test_mod.sample(5, decode = False, as_pandas = True, labels = sample_labels)
        out_enc_np = test_mod.sample(5, decode = False, as_pandas = False, labels = sample_labels)


        with self.subTest(msg="Check column names (decoded | pandas)"):
            self.assertEqual(
                out_dec_pd.columns.tolist(),
                ['x2','bin1','y','x1','cat_lat1','cat_lat2','cat','x3']
            )

        with self.subTest(msg="Check decoding (decoded | pandas)"):
            uniq_vals = out_dec_pd['cat_lat1'].unique()
            self.assertTrue(set(uniq_vals).issubset(set(['x','y','z'])))

        with self.subTest(msg="Check softmax (encoded | pandas)"):
            cat_lat2 = out_enc_pd[['cat_lat2_q','cat_lat2_r']]
            self.assertEqual([round(num,5) for num in cat_lat2.sum(axis=1).tolist()], [1.00000 for i in range(5)])

        with self.subTest(msg="Check softmax (encoded | numpy)"):
            cat_lat1 = out_enc_np[:,4:7]
            self.assertEqual([round(num,5) for num in cat_lat1.sum(axis=1).tolist()], [1.00000 for i in range(5)])

        

        


