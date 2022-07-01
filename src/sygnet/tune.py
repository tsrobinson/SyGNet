from .requirements import *
from .dataloaders import GeneratedData
from .interface import SygnetModel

import random
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

def tune(
    parameter_dict,
    data,
    test_func, 
    runs,
    model_opts = {},
    fit_opts = {},
    test_opts = {},
    mode = "wgan",
    k = 5,
    tuner = "random",
    n = 1000,
    epochs = 50,
    seed = 89,
    device = 'cpu'):
    """Find optimal values for a SyGNet model 

        Args:
            parameter_dict (dict): A dictionary of hyperparameter arguments and list of values to try. Currently, this function supports tuning the following parameters: `hidden_nodes`, `dropout_p`,`layer_norms`,`relu_leak`, `batch_size,`learning_rate`, and `adam_betas`"
            data (str or pd.DataFrame): Real data used to train GAN, can be a filepath or Pandas DataFrame
            test_func (func): Function used to assess the quality of the model. This can be any custom function, but must accept a `model` argument which will be a trained SygnetModel()
            runs (int): Number of hyperparameter combinations to try
            model_opts (dict): Dictionary of fixed arguments to pass to SygnetModel().
            fit_opts (dict): Dictionary of fixed arguments to pass to SygnetModel.fit().
            test_opts (dict): Dictionary of fixed arguments to pass to `test_func`
            mode (str): One of ["basic","wgan","cgan"]. Determines whether to use basic GAN, Wasserstein loss, or Conditional GAN training method (default = "wgan").
            k (int): Number of folds for adapted k-fold validation
            tuner (str): Placeholder argument for type of hyperparameter sampling to conduct -- currently only random sampling is supported
            n (int): Number of synthetic observations to draw and pass to `test_func`
            epochs (int): Number of epochs to train each model for
            seed (int): Random seed
            device (str): Whether to train model on the "cpu" (default) or "cuda" (i.e. GPU-training).

        Returns:
            pd.DataFrame of hyperparameter tuning results, with k observations per sample of hyperparameter values

    """

    logger.warning(
        "THIS FUNCTION IS STILL IN DEVELOPMENT. \
        Only 'wgan' modelling has been implemented thus far, \
            and all hyperparameter searches will use random \
                sampling rather than an exhaustive grid seach"
    )

    torch.manual_seed(seed)
    random.seed(seed)

    if mode != "wgan":
        return None

    if type(parameter_dict) is not dict:
        logger.error("`parameter_dict` must be a dictionary with hyperparameter arguments as keys and lists of options to try as values. \n \
            Tunable hyperparameters across sygnet are currently: \n \
                \t SygnetModel: `hidden_nodes`, `dropout_p`,`layer_norms`,`relu_leak`, \n \
                \t .fit(): `batch_size,`learning_rate`, and `adam_betas`"
                )

    model_hyps = ['hidden_nodes','dropout_p','relu_leak','layer_norms']
    fit_hyps = ['batch_size','learning_rate','adam_betas']

    model_dict = dict((k, parameter_dict[k]) for k in model_hyps if k in parameter_dict)
    fit_dict = dict((k, parameter_dict[k]) for k in fit_hyps if k in parameter_dict)

    tuning_results = []

    for i in range(runs):

        model_dict_chosen = {k: random.choice(v) for k,v in model_dict.items()}
        fit_dict_chosen = {k: random.choice(v) for k,v in fit_dict.items()}

        kf = KFold(n_splits=k)
        kf.get_n_splits(data)

        k_count = 0

        for train_idx, kth_idx in kf.split(data):

            sygnet_model = SygnetModel(**model_dict_chosen, **model_opts, mode = mode)

            sygnet_model.fit(
                data.iloc[train_idx,:],
                **fit_dict_chosen,
                **fit_opts,
                epochs = epochs,
                device = device)

            k_out = test_func(model = sygnet_model, **test_opts)

            tuning_results.append([i, k_count, k_out] + list(model_dict_chosen.values()) + list(fit_dict_chosen.values()))

            k_count += 1
        
    tuning_results = pd.DataFrame(tuning_results)
    tuning_results.columns = ["it", "k-fold", "fun_out"] +  list(model_dict_chosen.keys()) + list(fit_dict_chosen.keys())
    
    return tuning_results


def critic_loss(model):
    """Helper function to extract W-loss 

        Args:
            model (func): Critic model

        Returns:
            critic loss from final batch-iteration of training

    """

    return model.disc_losses[-1]