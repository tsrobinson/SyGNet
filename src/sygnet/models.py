from .requirements import *
from .blocks import *

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
            if self.funcs[number] == 'identity':
                mixed_out.append(self.identity(torch.index_select(input, 1, index_.type(torch.int32).to(self.device))))
            elif self.funcs[number] == 'softmax':
                mixed_out.append(nn.functional.gumbel_softmax(torch.index_select(input, 1, index_.type(torch.int32).to(self.device)),tau=0.66, hard=False, dim=1))
            else:
              pass

        col_order = torch.argsort(torch.cat(self.indices))
        return torch.cat(mixed_out,1)[:,col_order]


class Generator(nn.Module):
    """Generator class for GAN network

    Args:
        input_size (int): The number of input nodes
        output_size (int): The number of output nodes (which may not equal input_size if using a CGAN architecture)
        n_blocks (int): The number of hidden layer blocks
        hidden_nodes (int): The number of nodes in each hidden layer of the generator network
        mixed_activation (boolean): Whether to use a mixed activation function final layer (default = True). If set to false, categorical and binary columns will not be properly transformed in generator output.
        
        attention (boolean): Whether to use a multi-headed attention model (default = True, if False then the architecture is a "SyGNet-LN1" feed-forward model)
        n_heads (int): The number of heads to include in attention block (default = None). Not required when attention = False.
        mixed_act_indices (list): Formatted list of continuous, positive continuous, binary, and softmax column indices (default = None).
        mixed_act_funcs (list): List of functions corresponding to elements of mixed_act_indices (default = None).
        dropout_p (float): The proportion of hidden nodes to be dropped randomly during training
        relu_alpha (float): The negative slope parameter used to construct hidden-layer ReLU activation functions (default = 0.01).
        device (str): Either 'cuda' or 'cpu', used to correctly initialise mixed activation layer (default = 'cpu')

    Attributes:
        output_size (int): The number of output nodes
        node_sizes (list): A list of node sizes per layer of the network
        dropout_p (float): The proportion of hidden nodes to be dropped randomly during training
        relu_alpha (float): The negative slope parameter used to construct hidden-layer ReLU activation functions
        out (nn.Module): The final activation function, either of class nn.Identity() or _MixedActivation()

    """

    def __init__(
        self, 
        input_size, 
        output_size,
        n_blocks,
        hidden_nodes,
        mixed_activation, 
        attention = True,
        n_heads = None,
        mix_act_indices = None, 
        mix_act_funcs = None, 
        dropout_p = 0.2, 
        relu_alpha = 0.01, 
        device = 'cpu',
        ):
        super(Generator, self).__init__()
        self.output_size = output_size # Generalization to allow for CGAN
        
        self.n_blocks = n_blocks
        self.hidden_nodes = hidden_nodes

        if dropout_p < 0 or dropout_p > 1:
            logger.error("dropout_p must be a real number in the range [0,1]")
        else:
            self.dropout_p = dropout_p
    
        self.relu_alpha = relu_alpha

        self.lin_in = nn.Linear(input_size, hidden_nodes, bias=True)

        if attention:
            self.n_heads = n_heads if n_heads is not None else 8
            self.blocks = nn.Sequential(
                *[LgBlock(n_heads=self.n_heads, n_lin = self.hidden_nodes, d_p = self.dropout_p) for _ in range(self.n_blocks)]
            )

        else:
            self.blocks = nn.Sequential(
                *[gLN1(n_lin = self.hidden_nodes, 
                       d_p = self.dropout_p,
                       r_a = self.relu_alpha) for i in range(n_blocks - 1)])
            
        self.lin_out = nn.Linear(hidden_nodes, output_size, bias=True)
        
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
        logger.debug("GENERATOR: forward pass")
        x = self.lin_in(x)
        x = self.blocks(x)
        x = self.lin_out(x)
        logger.debug("GENERATOR: output activation")
        x = self.out(x)
        return x

class Critic(nn.Module):
    """Critic class for GAN network

    Args:
        input_size (int): The number of input nodes
        n_blocks (int): The number of hidden blocks to include
        hidden_nodes (int): The number of nodes per layer in the *hidden* linear layers
        dropout_p (float): The proportion of hidden nodes to be dropped randomly during training
        relu_alpha (float): The negative slope parameter used to construct hidden-layer ReLU activation functions (default = 0.1; note: this default is an order larger than torch default.)

    Attributes:
        n_blocks (int): Number of hidden blocks
        lin1 (torch.nn.ModuleList): Input layer to the network
        blocks (torch.nn.Sequential): A torch-formatted list of linear blocks in the network (linear -> dropout -> leakReLU)
        dropout_p (float): The proportion of hidden nodes to be dropped randomly during training
        relu_alpha (float): The negative slope parameter used to construct hidden-layer ReLU activation functions
        out (nn.Module): The final output layer (the critic scores)

    """

    def __init__(self, input_size, n_blocks, hidden_nodes, dropout_p = 0.2, relu_alpha = 0.01):
        super(Critic, self).__init__()
        self.n_blocks = n_blocks
        self.hidden_nodes = hidden_nodes
        self.relu_alpha = relu_alpha
        
        if dropout_p < 0 or dropout_p > 1:
            logger.error("dropout_p must be a real number in the range [0,1]")
        else:
            self.dropout_p = dropout_p

        # Layers
        self.lin1 = nn.Linear(input_size, hidden_nodes, bias=True)
        self.blocks = nn.Sequential(
            *[LcBlock(n_lin = self.hidden_nodes,
                      d_p = self.dropout_p, 
                      r_a = self.relu_alpha) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden_nodes, 1, bias=True)
        

    def forward(self, x):
        """Forward pass method

        Args:
            x (Tensor): Input data

        Returns:
            x (Tensor): The score of each input observation's 'realness'.

        """
        logger.debug("DISCRIMINATOR FORWARD")
        x = self.lin1(x)
        x = self.blocks(x)
        x = self.out(x)
        
        return x

class ConditionalWrapper(nn.Module):
    def __init__(self, latent_size, label_size, main_network, relu_alpha = 0.1):
        super().__init__()
        self.conditional = True
        self.latent_size = latent_size
        self.label_size = label_size
        self.combiner = nn.Sequential(
            nn.Linear(latent_size+label_size, latent_size),
            nn.LeakyReLU(negative_slope=relu_alpha),
            # nn.Linear(latent_size, latent_size), # Think this is excessive as it can be controlled through the regular network
            # nn.LayerNorm(latent_size),
            # nn.LeakyReLU(negative_slope=relu_alpha)
        )
        self.net = main_network

    def forward(self, x, labels):
        x_comb = torch.cat([x, labels], dim=1)
        return self.net(self.combiner(x_comb))
