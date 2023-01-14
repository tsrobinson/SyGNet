from .requirements import *
from .dataloaders import GeneratedData
from .interface import SygnetModel

import random
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

def critic_loss(model, **kwargs):
    """Helper function to extract W-loss from trained model 

        Args:
            model (func): Critic model
            **kwargs: Additional arguments that will have no effect on output

        Returns:
            critic loss from final batch-iteration of training

    """

    return model.disc_losses[-1]

def tune(
    parameter_dict,
    data,
    runs,
    epochs = 50,
    checkpoints = 1,
    test_func = critic_loss, 
    model_opts = {},
    fit_opts = {},
    test_opts = {},
    mode = "wgan",
    k = 1,
    tuner = "random",
    seed = 89,
    device = 'cpu'):
    """Find optimal values for a SyGNet model 

        Args:
            parameter_dict (dict): A dictionary of hyperparameter arguments and list of values to try. Currently, this function supports tuning the following parameters: `hidden_nodes`, `dropout_p`,`layer_norms`,`relu_leak`, `batch_size,`learning_rate`, and `adam_betas`"
            data (str or pd.DataFrame): Real data used to train GAN, can be a filepath or Pandas DataFrame
            runs (int): Number of hyperparameter combinations to try
            epochs (int): Total number of epochs to train each model for
            checkpoints (int): Number of times to assess the model within each run. The test function will be run every epochs//checkpoints iterations.
            test_func (func): Function used to assess the quality of the model (default is the training critic loss, which has known convergence properties). This argument can be any custom function, but must accept a `model` argument which will be a trained SygnetModel()
            model_opts (dict): Dictionary of fixed arguments to pass to SygnetModel().
            fit_opts (dict): Dictionary of fixed arguments to pass to SygnetModel.fit().
            test_opts (dict): Dictionary of fixed arguments to pass to `test_func`
            mode (str): One of ["basic","wgan","cgan"]. Determines whether to use basic GAN, Wasserstein loss, or Conditional GAN training method (default = "wgan").
            k (int): Number of folds for k-fold procedures (default = 1, i.e. no K-fold validation). Users writing custom test functions can access the holdout data indices using `test_idx`
            tuner (str): Placeholder argument for type of hyperparameter sampling to conduct -- currently only random sampling is supported
            seed (int): Random seed
            device (str): Whether to train model on the "cpu" (default) or "cuda" (i.e. GPU-training).

        Returns:
            pd.DataFrame of hyperparameter tuning results

    """

    logger.warning(
        "This function is still in development. Only 'wgan' and 'cgan' modelling has been implemented thus far, and all hyperparameter searches will use random sampling rather than an exhaustive grid seach"
    )

    torch.manual_seed(seed)
    random.seed(seed)

    if mode not in ["wgan", "cgan"]:
        return None

    if mode == "cgan" and 'cond_cols' not in fit_opts:
        logger.warning(
            "Since you are using Conditional arcgitecture, you need to specify conditional columns as 'cond_cols' in fit_opts dictionary. Example: fit_opts = {'save_model': False, 'cond_cols' : ['name']}"
        )

    if mode == "cgan" and not isinstance(fit_opts['cond_cols'],list):
        raise Exception("Conditional columns 'cond_cols' must be a list!")

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

        if k > 1:
            kf = KFold(n_splits=k)
            kf.get_n_splits(data)
        else:
            kf = KFold(n_splits =2)
            kf.get_n_splits(data)
        
        k_count = 0

        for train_idx, _ in kf.split(data):

            if k == 1 and k_count == 1:
                continue
            
            sygnet_model = SygnetModel(**model_dict_chosen, **model_opts, mode = mode)
            train_epochs = epochs//checkpoints
            
            for c in range(checkpoints):
                sygnet_model.fit(
                    data = data.iloc[train_idx,:] if k > 1 else data,
                    **fit_dict_chosen,
                    **fit_opts,
                    epochs = train_epochs,
                    device = device)
                current_epoch = (c+1)*train_epochs
                k_out = test_func(model = sygnet_model, **test_opts)
                
                tuning_results.append([i, k_count, current_epoch, k_out] + list(model_dict_chosen.values()) + list(fit_dict_chosen.values()))

            k_count += 1
        
    tuning_results = pd.DataFrame(tuning_results)
    tuning_results.columns = ["it", "fold","epochs","fun_out"] +  list(model_dict_chosen.keys()) + list(fit_dict_chosen.keys())
    
    return tuning_results


