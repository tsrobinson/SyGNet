from .requirements import *
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

logger = logging.getLogger(__name__)

class GeneratedData(Dataset):
    """Formats processed data for GAN training, optionally using the conditional infrastructure

    Args:
        real_data (str or pd.DataFrame): If a string is supplied, the file location of the data. Otherwise, a pd.DataFrame of training data
        conditional (bool): Whether or not to format data for conditional GAN architecture (default = False)
        cond_cols (list of colnames): If conditional is True, the column names of the real data that should serve as the conditional labels in the model  

    Attributes:
        n_samples (int): Number of observations in the training data
        x (Tensor): Training data formatted for PyTorch modelling
        x_indx, x_funcs (list): Indices and output activation functions, respectively, for datatypes in x
        labels (Tensor): Conditional training labels formatted for PyTorch modelling
        x_OHE, labels_OHE (obj): OneHotEncoding model objects, used to inverse transform data
        colnames: Names of Pandas DataFrame columns
        
    """
   
    def __init__(self, real_data, conditional = False, cond_cols= None):
        # Allow loading of data from file or memory
        if type(real_data) == str:
            try:
                data_in = pd.read_csv(real_data)
            except:
                logger.error(f"Unable to load data from location {real_data}. Please check path or supply object of class pd.DataFrame")
        else:
            data_in = real_data.copy()

        self.n_samples = data_in.shape[0]

        if conditional:

            # Separate data
            cond_labels = data_in.loc[:,cond_cols]
            data_in.drop(cond_cols, axis = 1, inplace = True)
            
            # Process latent data
            self.x, self.x_indxs, self.x_funcs, self.x_transformers, self.colnames = _preprocess_df(data_in)
            self.x = torch.from_numpy(self.x)

            # Process conditional labels (no need to save funcs as won't be fed to activation)
            self.labels,_,_,self.labels_transformers, label_names = _preprocess_df(cond_labels)
            self.labels = torch.from_numpy(self.labels)
            self.colnames += label_names

        else:
            self.x, self.x_indxs, self.x_funcs, self.x_transformers, self.colnames = _preprocess_df(data_in)
            self.x = torch.from_numpy(self.x)
            self.labels = torch.ones(self.n_samples, 1)
        
    def __getitem__(self, index):
        return self.x[index], self.labels[index]

    def __len__(self):
        return self.n_samples


def _preprocess_df(df):
    '''
    Sort and arrange columns for managing mixed activation
    Args:
        df(pd.Dataframe): The input data
    Returns:
        df (np.array)
        col_idx (list): Tuples with 'column name' and list of one-hot indices for that column plus all numeric columns)
        col_fs (list): List of functions for each column in data
        transformers (tuple): OHE object and min-max scaler for inverse data transformation
        df_cols (list): List of column names after column sorting but before one-hot encoding
    '''
    # 1. get categorical colum names
    num_type = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']  # numeric columns
    str_type = 'O'  # select desired data type: 'O' - string
    categorical_cols = []
    numeric_cols = []
    dtypes = df.dtypes.to_dict()

    for colname, data_type in dtypes.items():
        if data_type == str_type:
            categorical_cols.append(colname)
        elif data_type in num_type:
          numeric_cols.append(colname)
        else:
            pass
    
    OHE = OneHotEncoder(sparse_output=False)
    scaler = MinMaxScaler()
    # fill missing categorical columns as nan
    df_cat = df[categorical_cols].fillna('nan')
    # OHe transform
    df_cat = OHE.fit_transform(df_cat)
    df_num = df.drop(categorical_cols, axis=1)
    # fill missing numeric values
    df_num = df_num.fillna(df_num.median())
    
    # get ordered list of column names
    df_cols = df_num.columns.tolist() + categorical_cols
    if df_num.shape[1] > 0:
        df_num = scaler.fit_transform(df_num)
    transformers = (OHE, scaler)
    
    # 3. finding idx for each original categorical column
    col_idx, col_fs = [], []
    
    # Numeric cols idx
    if len(numeric_cols) != 0:
        col_idx_tensor = torch.Tensor([c for c in range(len(numeric_cols))])
        col_idx.append(col_idx_tensor)
        col_fs.append('identity')

    # Categorical cols idx
    n_numeric = df_num.shape[1]
    cat_current_count = 0
    for var in OHE.categories_:
        one_hot_cols = var.tolist()
        start_idx = n_numeric + cat_current_count
        col_idx_tensor = torch.Tensor([i for i in range(start_idx, start_idx + len(one_hot_cols))])
        cat_current_count += len(one_hot_cols)
        col_idx.append(col_idx_tensor)
        col_fs.append('softmax')

    df = np.concatenate((df_num, df_cat), axis = 1, dtype=np.float32) 

    return df, col_idx, col_fs, transformers, df_cols 

def _ohe_colnames(OHE):
    cat_cols = []
    for i in range(OHE.n_features_in_):
        for j in OHE.categories_[i]:
            cat_cols.append(OHE.feature_names_in_[i]+"_"+j)
    return cat_cols