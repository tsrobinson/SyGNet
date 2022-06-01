from .sygnet_requirements import *

logger = logging.getLogger(__name__)

class _MixedActivation(nn.Module):
    """Custom activation layer that applies separate functionals for continuous, binary, and categorical data.

    Args:
        indices (list of lists): Positions of data types. 
        funcs (list of strs): Corresponding list of activation functions       
        device (str): Model device for use in forward()

    Notes:
        Relies on _preprocessing() from sygnet_dataloaders to pre-sort and format data.

    """

    __constants__ = ['indices', 'funcs', 'device']
    inplace: list
    funcs: list
    def __init__(self, indices, funcs, device):
        super(_MixedActivation, self).__init__()
        if any(item not in ['identity','relu','sigmoid','softmax'] for item in set(funcs)):
            logger.error("Cannot construct output layer: unrecognised mixed activation functional")
        self.indices = indices
        self.funcs = funcs
        self.identity = nn.Identity()
        self.device = device
        
    def forward(self, input: Tensor) -> Tensor:
        mixed_out = []
        for number, index_ in enumerate(self.indices):
            if self.funcs[number] == 'softmax':
                mixed_out.append(nn.functional.softmax(torch.index_select(input, 1, index_.type(torch.int32).to(self.device)), dim=1))
            elif self.funcs[number] == 'relu':
                mixed_out.append(nn.functional.relu(torch.index_select(input, 1, index_.type(torch.int32).to(self.device))))
            elif self.funcs[number] == 'sigmoid':
                mixed_out.append(torch.sigmoid(torch.index_select(input, 1, index_.type(torch.int32).to(self.device))))
            elif self.funcs[number] == 'identity':
                mixed_out.append(self.identity(torch.index_select(input, 1, index_.type(torch.int32).to(self.device))))

        col_order = torch.argsort(torch.cat(self.indices))
        return torch.cat(mixed_out,1)[:,col_order]


class Generator(nn.Module):
    """Generator class for GAN network

    Args:
        input_size (int): The number of input nodes
        hidden_sizes (list of ints): A list of ints, containing the number of nodes in each hidden layer of the generator network
        output_size (int): The number of output nodes
        mixed_activation (boolean): Whether to use a mixed activation function final layer (default = True). If set to false, categorical and binary columns will not be properly transformed in generator output.
        mixed_act_indices (list): Formatted list of continuous, positive continuous, binary, and softmax column indices (default = None).
        mixed_act_funcs (list): List of functions corresponding to elements of mixed_act_indices (default = None).
        dropout_p (float): The proportion of hidden nodes to be dropped randomly during training
        relu_alpha (float): The negative slope parameter used to construct hidden-layer ReLU activation functions (default = 0.1; note: this default is an order larger than torch default.)
        layer_norm (boolean): Whether to include layer normalization in network (default = True)
        device (str): Either 'cuda' or 'cpu', used to correctly initialise mixed activation layer (default = 'cpu')

    Attributes:
        output_size (int): The number of output nodes
        node_sizes (list): A list of node sizes per layer of the network
        dropout_p (float): The proportion of hidden nodes to be dropped randomly during training
        relu_alpha (float): The negative slope parameter used to construct hidden-layer ReLU activation functions
        linears (torch.nn.ModuleList): A torch-formatted list of linear layers in the network
        dropouts (torch.nn.ModuleList): A torch-formatted list of dropout layers in the network
        hidden_acts (torch.nn.ModuleList): A torch-formatted list of leaky-ReLU activation functions
        layer_norms (torch.nn.ModuleList): A torch-formatted list of LayerNorm functions
        out (nn.Module): The final activation function, either of class nn.Identity() or _MixedActivation()

    """

    def __init__(
        self, 
        input_size, 
        hidden_sizes, 
        output_size, 
        mixed_activation, 
        mix_act_indices = None, 
        mix_act_funcs = None, 
        dropout_p = 0.2, 
        layer_norm = True, 
        relu_alpha = 0.1, 
        device = 'cpu'
        ):
        super(Generator, self).__init__()
        self.output_size = output_size
        self.node_sizes = [input_size] + hidden_sizes + [output_size]

        self.linears = nn.ModuleList(
            [nn.Linear(self.node_sizes[i-1], self.node_sizes[i]) for i in range(1, len(self.node_sizes))]
            )

        if layer_norm:
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(self.node_sizes[i+1]) for i in range(len(self.node_sizes)-2)]
                )
        else:
            self.layer_norms = nn.ModuleList()
        
        self.relu_alpha = relu_alpha
        self.hidden_acts = nn.ModuleList(
            [nn.LeakyReLU(negative_slope = self.relu_alpha) for i in range(len(self.node_sizes)-2)]
            )

        if dropout_p < 0 or dropout_p > 1:
            logger.error("dropout_p must be a real number in the range [0,1]")
        elif dropout_p == 0:
            self.dropout_p = dropout_p
            self.dropouts = nn.ModuleList()
        else:
            self.dropout_p = dropout_p
            self.dropouts = nn.ModuleList(
                [nn.Dropout(p = self.dropout_p) for i in range(len(self.node_sizes)-2)]
                )
        
        if mixed_activation:
            self.out = _MixedActivation(mix_act_indices, mix_act_funcs, device)
        else:
            self.out = nn.Identity()
            logger.warning("Not using mixed activation function -- generated data may not conform to real data if it contains categorical columns.")

    def forward(self, x):
        """Forward pass method for generator network

        Args:
            x (Tensor): Input data

        Returns:
            x (Tensor): Output data

        """
        logger.debug("GENERATOR FORWARD")
        for i in range(len(self.linears)):
            logger.debug("Layer "+str(i)+": Linear")
            x = self.linears[i](x)
            if i < len(self.layer_norms):
                logger.debug("Layer "+str(i)+": LN")
                x = self.layer_norms[i](x)
            if i < len(self.hidden_acts):
                logger.debug("Layer "+str(i)+": Leaky ReLU")
                x = self.hidden_acts[i](x)
            if i < len(self.dropouts):
                logger.debug("Layer "+str(i)+": Dropout")
                x = self.dropouts[i](x)
        logger.debug("Output activation")
        x = self.out(x)
        return x

class Discriminator(nn.Module):
    """Discriminator class for GAN network

    Args:
        input_size (int): The number of input nodes
        hidden_sizes (list of ints): A list of ints, containing the number of nodes in each hidden layer of the discriminator network
        dropout_p (float): The proportion of hidden nodes to be dropped randomly during training
        relu_alpha (float): The negative slope parameter used to construct hidden-layer ReLU activation functions (default = 0.1; note: this default is an order larger than torch default.)
        layer_norm (boolean): Whether to include layer normalization in network (default = True)

    Attributes:
        node_sizes (list): A list of node sizes per layer of the network
        linears (torch.nn.ModuleList): A torch-formatted list of linear layers in the network
        dropout_p (float): The proportion of hidden nodes to be dropped randomly during training
        relu_alpha (float): The negative slope parameter used to construct hidden-layer ReLU activation functions
        linears (torch.nn.ModuleList): A torch-formatted list of linear layers in the network
        dropouts (torch.nn.ModuleList): A torch-formatted list of dropout layers in the network
        hidden_acts (torch.nn.ModuleList): A torch-formatted list of leaky-ReLU activation functions
        layer_norms (torch.nn.ModuleList): A torch-formatted list of LayerNorm functions
        out (nn.Module): The final activation function, always nn.Sigmoid()

    """

    def __init__(self, input_size, hidden_sizes, dropout_p = 0.2, layer_norm = True, relu_alpha = 0.1):
        super(Discriminator, self).__init__()
     
        self.node_sizes = [input_size] + hidden_sizes + [1]

        self.linears = nn.ModuleList(
            [nn.Linear(self.node_sizes[i-1], self.node_sizes[i]) for i in range(1, len(self.node_sizes))]
            )

        if layer_norm:
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(self.node_sizes[i+1]) for i in range(len(self.node_sizes)-2)]
                )
        else:
            self.layer_norms = nn.ModuleList()
        
        self.relu_alpha = relu_alpha
        self.hidden_acts = nn.ModuleList(
            [nn.LeakyReLU(negative_slope = self.relu_alpha) for i in range(len(self.node_sizes)-2)]
            )        

        if dropout_p < 0 or dropout_p > 1:
            logger.error("dropout_p must be a real number in the range [0,1]")
        elif dropout_p == 0:
            self.dropout_p = dropout_p
            self.dropouts = nn.ModuleList()
        else:
            self.dropout_p = dropout_p
            self.dropouts = nn.ModuleList(
                [nn.Dropout(p = self.dropout_p) for i in range(len(self.node_sizes)-2)]
                )

        self.out = nn.Sigmoid()

    def forward(self, x):
        """Forward pass method

        Args:
            x (Tensor): Input data

        Returns:
            x (Tensor): If using a Discriminator, the probability of each input observation being real (1) or fake (0). 
                        If using a Critic, the score of each input observation's 'realness'.

        """

        logger.debug("DISCRIMINATOR FORWARD")
        for i in range(len(self.linears)):
            logger.debug("Layer "+str(i)+": Linear")
            x = self.linears[i](x)
            # Pass FC layer through LN
            if i < len(self.layer_norms):
                logger.debug("Layer "+str(i)+": LN")
                x = self.layer_norms[i](x)
            # Then through leaky ReLU
            if i < len(self.hidden_acts):
                logger.debug("Layer "+str(i)+": Leak ReLU")
                x = self.hidden_acts[i](x)
            # Then apply dropout    
            if i < len(self.dropouts):
                logger.debug("Layer "+str(i)+": Dropout")
                x = self.dropouts[i](x)
        logger.debug("Output activation")
        x = self.out(x)     
        return x

class Critic(Discriminator):
    """Critic class for WGAN network

    Args:
        input_size (int): The number of input nodes
        hidden_sizes (list of ints): A list of ints, containing the number of nodes in each hidden layer of the critic network
        dropout_p (float): The proportion of hidden nodes to be dropped randomly during training
        relu_alpha (float): The negative slope parameter used to construct hidden-layer ReLU activation functions (default = 0.1; note: this default is an order larger than torch default.)
        layer_norm (boolean): Whether to include layer normalization in network (default = True)

    Attributes:
        node_sizes (list): A list of node sizes per layer of the network
        linears (torch.nn.ModuleList): A torch-formatted list of linear layers in the network
        dropout_p (float): The proportion of hidden nodes to be dropped randomly during training
        relu_alpha (float): The negative slope parameter used to construct hidden-layer ReLU activation functions
        linears (torch.nn.ModuleList): A torch-formatted list of linear layers in the network
        dropouts (torch.nn.ModuleList): A torch-formatted list of dropout layers in the network
        hidden_acts (torch.nn.ModuleList): A torch-formatted list of leaky-ReLU activation functions
        layer_norms (torch.nn.ModuleList): A torch-formatted list of LayerNorm functions
        out (nn.Module): The final activation function, always nn.Identity()

    """

    def __init__(self, input_size, hidden_sizes, dropout_p = 0.2, layer_norm = True, relu_alpha = 0.1):
        super().__init__(input_size, hidden_sizes, dropout_p , layer_norm , relu_alpha)
        self.out = nn.Identity()

class ConditionalWrapper(nn.Module):
    def __init__(self, latent_size, label_size, main_network, relu_alpha = 0.1):
        super().__init__()
        self.conditional = True
        self.latent_size = latent_size
        self.label_size = label_size
        self.combiner = nn.Sequential(
            nn.Linear(latent_size+label_size, latent_size),
            nn.LeakyReLU(negative_slope=relu_alpha),
            nn.Linear(latent_size, latent_size),
            nn.LayerNorm(latent_size),
            nn.LeakyReLU(negative_slope=relu_alpha)
        )
        self.net = main_network

    def forward(self, x, labels):
        x_comb = torch.cat([x, labels], dim=1)
        return self.net(self.combiner(x_comb))
