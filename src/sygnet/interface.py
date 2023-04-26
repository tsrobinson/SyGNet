## USER INTERFACE
import logging
import os.path
from pathlib import Path

from .requirements import *
from .models import Generator, Critic, ConditionalWrapper
from .train import train
from .dataloaders import GeneratedData, _ohe_colnames


logger = logging.getLogger(__name__)

class SygnetModel:

    def __init__(
        self,
        # Model type 
        mode,
        attention = True,

        # Generator and Discriminator options
        n_blocks = 2,
        hidden_nodes = 256, 
        dropout_p = 0.2, 
        relu_leak = 0.01,
        n_heads = None,
        mixed_activation = True
        ):
        """SyGNet model object

        Args:
            mode (str): One of ["wgan","cgan"]. Determines whether to use Wasserstein loss or Conditional GAN training method (default = "wgan").
            attention (bool): Whether to use self-attention within the generator model (default = True).
            n_blocks (int, or [int, int]): The number of hidden blocks to use in the generator and critic networks (default = 2). See notes for more details.
            hidden_nodes (int, or [int, int]): The number of nodes in each hidden layer of the generator and critic networks (default = 256). 
            dropout_p (float, or [float, float]): The proportion of hidden nodes to be dropped randomly during training from the generator and critic respectively (default = 0.2).
            relu_leak (float, or [float, float]): The negative slope parameters used to construct hidden-layer ReLU activation functions (default = 0.01)
            n_heads (int): The number of attention heads within the self-attention mechanism (default = None).
            mixed_activation (boolean): Whether to use a mixed activation function final layer for the generator (default = True). If set to false, categorical and binary columns will not be properly transformed in generator output.

        Notes: 
            hidden_nodes, dropout_p, and relu_leak can all accept list arguments of length 2, specifying parameters for the [generator, discriminator] respectively.
            Parameters for the generator and discriminator are set independently (with identical default settings). Both models are modified in-place.
            The hidden blocks of the network comprise different steps dependent on the architecture. At present, these are fixed by the `attention` argument. When attention = True, a single block comprises self attention -> batch normalisation -> leaky-ReLU activation. If attention = False, a single block comprises a linear layer -> layer normalisation -> leaky-ReLU -> dropout.  

        Attributes:
            mode (str):
            input_size (int): The number of input nodes. `None` until .fit() is called.
            label_size (int): The number of additional nodes for label columns when mode = "cgan". `None` until .fit() is called.
            output_size (int): The number of output nodes
            generator (nn.Module): The generator network (`None` until data fit). See `help(Generator)` for more information on internal attributes.
            mixed_activation (bool): Inclusion of a mixed activation layer
            discriminator (nn.Module): The discriminator/critic network (None until data fit). See `help(Discriminator)` or `help(Critic)` for more information on internal attributes.
            gen_hidden(list)
            disc_hidden(list)
            gen_dropout (float)
            disc_dropout (float)
            disc_lin (bool)
            gen_leak (float)
            disc_leak (float)
            data_encoders (list)
            gen_losses (list)
            disc_losses (list)

        """
        self.mode = mode
        self.attention = attention
        self.n_heads = n_heads

        self.input_size = None
        self.label_size = None
        self.output_size = None
        
        self.generator = None
        self.critic = None

        self.mixed_activation = mixed_activation
        self.data_encoders = None
        self.colnames = None

        self.num_idx = None

        self.gen_losses = []
        self.disc_losses = []

        # Check args
        if self.mode not in ['wgan','cgan']:
            logger.error("Argument `mode` must be one of 'wgan' or 'cgan'")

        if type(n_blocks) is int:
            self.gen_blocks = self.crit_blocks = n_blocks
        elif type(n_blocks) is list and len(n_blocks) == 2:
            self.gen_blocks, self.crit_blocks = n_blocks
        else:
            logger.error("Argument `n_blocks` must either be a float or a list/tuple of length 2")

        # Get hidden node sizes
        if type(hidden_nodes) is int:
            self.gen_hidden = self.crit_hidden = hidden_nodes
        elif type(hidden_nodes) is list and len(hidden_nodes) == 2:
            self.gen_hidden, self.crit_hidden = hidden_nodes
        else:
            logger.warning("Supplying separate lists for each model was removed in v0.0.9")
            logger.error("Argument `hidden_nodes` must either be a list of hidden layer sizes, or an integer.")

        # Get dropout proportions
        if type(dropout_p) is float:
            self.gen_dropout = self.crit_dropout = dropout_p
        elif type(dropout_p) is list and len(dropout_p) == 2:
            self.gen_dropout, self.crit_dropout = dropout_p
        else:
            logger.error("Argument `dropout_p` must either be a float or a list/tuple of length 2")

        # Get leaky alpha values
        if type(relu_leak) is float:
            self.gen_leak = self.crit_leak = relu_leak
        elif type(relu_leak) is list and len(relu_leak) == 2:
            self.gen_leak, self.crit_leak = relu_leak
        else:
            logger.error("Argument `relu_leak` must either be a float or a list/tuple of length 2.") 
    
    def fit(
        self, 
        data,
        cond_cols = None,
        epochs=10, 
        device='cpu',
        batch_size = None,
        learning_rates = 0.0001,
        lmbda = 10,
        use_tensorboard = False,
        save_model = False,
        save_loc = "",
        save_prefix = "sygnet_model_"
        ):
        """Fit the SyGNet model to the training data

        Args:
            data (str or pd.DataFrame): Real data used to train GAN, can be a filepath or Pandas DataFrame
            cond_cols (list of colnames): Column names that indicate conditioning variables
            epochs (int): Number of training epochs
            device (str): Either 'cuda' for GPU training, or 'cpu' (default='cpu')
            batch_size (int): Number of training observations per batch (default = None). If left at default, the batch size is set to 1/20th of the overall length of the data.
            learning_rates (float or [float, float]): The learning rates for the generator and critic AdamW optimizers
            lmbda (float): Scalar penalty term for applying gradient penalty as part of Wasserstein loss, only used in wgan and cgan modes
            use_tensorboard (boolean): If True, creates tensorboard output capturing key training metrics (default = True)
            save_model (bool): Whether or not to save the model after training (default = False)
            save_loc (str): If save_model is True, the filepath where the directory should be saved (default = current working directory). Note, on Windows users should use raw strings (i.e. r"C:..." to avoid backslash escape issues)
            save_prefix (str): File prefix for the saved model (default = "sygnet_model_"). The full filename will be save_prefix + "DDMMMYY_HHMM"

        Note: 
            The generator and discriminator/critic model are modified in-place

        Returns:
            None

        """

        ## Sort out the data:

        # Check args
        if cond_cols is not None and self.mode != "cgan":
            logger.warning(f"Conditional column indices supplied but model mode is set to '{self.mode}'. All columns in 'cond_cols'  will be synthesised.")
        elif cond_cols is None and self.mode == "cgan":
            logger.warning(f"Model mode is set to '{self.mode}' but no columns specified in 'cond_cols'. Switching to WGAN architecture.")
            self.mode = "wgan"

        if type(learning_rates) is float:
            learning_rates = [learning_rates, learning_rates]
        elif len(learning_rates) == 2:
            learning_rates = learning_rates
        else:
            logger.error("Learning rates argument must be a float or a list of two floats, for the generator and discriminator respectively.")

        # Set file path and static part of filename
        if save_model:
            if not isinstance(save_loc, str):
                logger.error("Argument `save_loc` must contain a directory path as an 'r' string. For example: save_loc = r'path/to/my/models/'")
                logger.error("Model will not be saved")
            elif save_loc == None:
                logger.error("Argument `save_loc` must contain a directory path as an 'r' string. For example: save_loc = r'path/to/my/models/'")
                logger.error("Model will not be saved")
            elif os.path.exists(save_loc) == False:
                logger.error("Argument `save_loc` must contain a directory path as an 'r' string. For example: save_loc = r'path/to/my/models/'")
                logger.error("Model will not be saved")
            else:
                logger.info("Model will be saved to: " + save_loc)
                filepath = Path(save_loc)

        # Convert data dependent on model type
        if self.mode != "cgan":
            torch_data = GeneratedData(real_data = data)
            self.input_size = self.output_size = torch_data.x.shape[1]
            # self.data_encoders = [torch_data.x_OHE]
            self.data_encoders = torch_data.x_transformers

        else:
            torch_data = GeneratedData(real_data = data, conditional = True, cond_cols = cond_cols)
            self.input_size = self.output_size = torch_data.x.shape[1]
            self.label_size = torch_data.labels.shape[1]
            # self.data_encoders = [torch_data.x_OHE, torch_data.labels_OHE]
            self.data_encoders = torch_data.x_transformers
            self.label_encoders = torch_data.labels_transformers

        self.colnames = torch_data.colnames
        self.num_idx = torch_data.x_indxs[0].long()
        
        ## Build the models (if not already built)

        if self.generator is None:      

            self.generator = Generator(

                input_size = self.input_size, 
                output_size = self.output_size,
                n_blocks = self.gen_blocks,
                hidden_nodes = self.gen_hidden,
                attention = self.attention,
                n_heads = self.n_heads,
                mixed_activation = self.mixed_activation,
                mix_act_indices=torch_data.x_indxs,
                mix_act_funcs=torch_data.x_funcs,
                dropout_p = self.gen_dropout,
                relu_alpha=self.gen_leak,
                device=device,
                )

            self.critic = Critic(
                input_size = self.input_size, 
                n_blocks = self.crit_blocks, 
                hidden_nodes = self.crit_hidden, 
                dropout_p = self.crit_dropout, 
                relu_alpha = self.crit_leak,
            )

            # Wrap conditional GANs
            if self.mode == "cgan":
                self.generator = ConditionalWrapper(latent_size = self.input_size, label_size = self.label_size, main_network = self.generator)
                self.critic = ConditionalWrapper(latent_size = self.input_size, label_size = self.label_size, main_network = self.critic)

        ## Train the models

        if batch_size is None:
            batch_size = int(np.floor(data.shape[0]/20))

        train_args = {
            'training_data' : torch_data, 
            'generator' : self.generator, 
            'critic' : self.critic,
            'epochs' : epochs, 
            'num_cols' : self.num_idx,
            'batch_size' : batch_size,
            'learning_rates' : learning_rates,
            'lmbda' : lmbda,
            'device' : device,
            'use_tensorboard' : use_tensorboard,
        }

        if self.mode == "wgan":
            g_loss, d_loss = train(**train_args, conditional = False)
        elif self.mode == "cgan":
            g_loss, d_loss = train(**train_args, conditional = True)
        
        self.gen_losses += g_loss
        self.disc_losses += d_loss
            
        if save_model:
            with open(filepath / (save_prefix + datetime.now().strftime("%d%b%y_%H%M")), 'wb') as f:
                pickle.dump(self, f)
        return None

    def sample(self, nobs, labels = None, file = None, decode = True, as_pandas = True, **kwargs):
        """Generate synthetic data 

        Args:
            generator_model (nn.Module): Generator model object
            nobs (nn.Module): Discriminator model object
            labels (pd.Dataframe): Array of labels that should have as many rows as 'nobs' argument. Only used if the Sygnet model has mode = "cgan" (default = None).
            file (str): File path location to save data. If a path is not provided, the data is only returned in memory (default = None).
            decode (bool): Whether to reverse one-hot encodings (default = True).
            as_pandas (bool): Whether to convert the GAN output from np.array to pd.DataFrame (default = True).
            
        Notes:
            We recommend keeping `as_pandas` as True to enable better tracking of variables. Since the training process reorders variables, values may be wrongly interpreted directly from a numpy array.
            Data is always returned in memory, regardless of whether a file path is provided.

        Returns:
            trained_output_df (pd.DataFrame): The generated synthetic data

        """

        if self.mode == "cgan":
            if labels is None:
                logger.error("No labels provided for CGAN. Users must specify conditions as pandas dataframe of length equal to `nobs`")
                raise ValueError("No labels provided for sampling from CGAN. ")
                # seed_labels = torch.rand(size=(nobs, self.label_size))
            else:
                if not isinstance(labels, pd.DataFrame):
                    try:
                        labels = pd.DataFrame(labels)
                    except:
                        logger.error("Labels was not provided as DataFrame: implicit conversion failed.")
                
                seed_latent = torch.rand(size=(nobs, self.generator.latent_size))

                labels_colnames_cat = self.label_encoders[0].feature_names_in_ if self.label_encoders[0].categories_ else []
                labels_cat = labels[labels_colnames_cat]

                labels_num = labels.drop(labels_colnames_cat, axis = 1)
                labels_colnames_num = labels_num.columns

                seed_labels = []
                if hasattr(self.label_encoders[1], 'n_features_in_'):
                    seed_labels.append(self.label_encoders[1].transform(labels_num))
                if hasattr(self.label_encoders[0], 'categories_'):
                    seed_labels.append(self.label_encoders[0].transform(labels_cat))

                seed_labels = torch.from_numpy(
                    np.concatenate(
                        seed_labels, 
                        axis=1
                    )
                ).float()
                
        else:
            seed_data = torch.rand(size=(nobs, self.generator.output_size))

        self.generator.eval()
        with torch.no_grad():
            device = next(self.generator.parameters()).device.type

            if self.mode == "cgan":
                seed_latent = seed_latent.to(device)
                seed_labels = seed_labels.to(device)
                synth_output = self.generator(seed_latent, seed_labels)                  
            else:
                seed_data = seed_data.to(device)
                synth_output = self.generator(seed_data)

            synth_output = synth_output.detach().to('cpu').numpy()
        
        logger.debug("Generated data")
        logger.debug(synth_output)

        self.generator.train()

        if decode:
            n_cat_vars = self.data_encoders[0].n_features_in_
            cat_names = [val for sublist in self.data_encoders[0].categories_ for val in sublist]
            n_cats = len(cat_names)
            if n_cat_vars > 0:
                logger.debug(f"{n_cats} categorical columns to transform for {n_cat_vars} categorical variables")
                synth_output = np.column_stack(
                    (synth_output[:,:-n_cats],
                    self.data_encoders[0].inverse_transform(synth_output[:,-n_cats:]))
                )
            else:
                synth_output = self.data_encoders[1].inverse_transform(synth_output)
            
            if self.mode == "cgan":
                synth_output = np.column_stack([synth_output, np.array(labels)])
                out_col_order = self.colnames[:-len(labels.columns)] + labels.columns.tolist()
        else:
            X_cat_cols = _ohe_colnames(self.data_encoders[0])
            labels_cat_cols =  []
            
            if self.mode == "cgan":
                synth_output = np.column_stack([synth_output, seed_labels]) 
                labels_cat_cols = _ohe_colnames(self.label_encoders[0])
                X_num_cols = self.colnames[:-(labels.shape[1]+self.data_encoders[0].n_features_in_)]
                out_col_order = X_num_cols + X_cat_cols + labels_colnames_num.tolist() + labels_cat_cols
            else:
                X_num_cols = self.colnames[:-self.data_encoders[0].n_features_in_]
                out_col_order = X_num_cols + X_cat_cols
                
        # Convert to pandas if required:
        if as_pandas:
            synth_output = pd.DataFrame(synth_output)
            if self.mode == "cgan":
                if decode:
                    synth_output.columns = out_col_order
                else:
                    synth_output.columns = out_col_order
            else:
                if decode:
                    synth_output.columns = self.colnames
                else:
                    synth_output.columns = out_col_order   
            
            logger.info("Data generated. Please check .columns attribute for order of variables.")

        if not decode and not as_pandas:
            logger.warning(f"Data generated as np.ndarray; variable order is {out_col_order}")
        
        # Check proper file name
        if file is not None:
            if file[-4:] != ".csv":
                file += ".csv"
            if as_pandas:
                synth_output.to_csv(path_or_buf = file, index=False)
            else:
                synth_output.tofile(file, sep=',')
            logger.info(f"Saved data to {file} using {'pandas' if as_pandas else 'numpy'}")

        return synth_output

    # Alias for sample to fit sklearn pipeline
    transform = sample

