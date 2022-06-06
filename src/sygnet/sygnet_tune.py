from .sygnet_requirements import *
from .sygnet_models import *
from .sygnet_train import *
from .sygnet_dataloaders import GeneratedData, _ohe_colnames
from .sygnet_interface import SygnetModel

import random
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

def tune(
    parameter_dict,
    data,
    test_fun, 
    runs,
    model_opts = {},
    fit_opts = {},
    test_opts = {},
    mode = "wgan",
    k = 5,
    tuner = "random",
    n = 1000,
    epochs = 50,
    seed = 89):

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
        logger.error("`parameter_dict` must be a dictionary with hyperparameters as keys and lists of options to try as values. \n \
            Tunable hyperparameters across sygnet are currently: \n \
                \t SygnetModel: `hidden_layers`, `dropout_p`,`layer_norms`,`relu_leak`, \n \
                \t .fit(): `batch_size,`learning_rate`, and `adam_betas`"
                )

    model_hyps = ['hidden_layers','dropout_p','relu_leak','layer_norms']
    fit_hyps = ['batch_size','learning_rate','adam_betas']

    model_dict = dict((k, parameter_dict[k]) for k in model_hyps if k in parameter_dict)
    fit_dict = dict((k, parameter_dict[k]) for k in fit_hyps if k in parameter_dict)

    tuning_results = []

    for i in range(runs):

        model_dict_chosen = {k: random.choice(v) for k,v in model_dict.items()}
        fit_dict_chosen = {k: random.choice(v) for k,v in fit_dict.items()}

        kf = KFold(n_splits=k)
        kf.get_n_splits(data)

        for train_idx, kth_idx in kf.split(data):

            sygnet_model = SygnetModel(**model_dict_chosen, **model_opts)

            sygnet_model.fit(
                data.iloc[train_idx,:],
                **fit_dict_chosen,
                **fit_opts,
                epochs = epochs)

            synth_data = sygnet_model.sample(n)

            k_out = test_fun(data = synth_data, **test_opts)

            tuning_results.append([i, kth_idx, k_out] + list(model_dict_chosen.values()) + list(fit_dict_chosen.values()))
        
    tuning_results = pd.DataFrame(tuning_results)
    tuning_results.columns = ["it", "k-fold", "fun_out"] +  list(model_dict_chosen.keys()) + list(fit_dict_chosen.keys())
    
    return tuning_results


