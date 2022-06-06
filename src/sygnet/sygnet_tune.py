from .sygnet_requirements import *
from .sygnet_models import *
from .sygnet_train import *
from .sygnet_dataloaders import GeneratedData, _ohe_colnames
from .sygnet_interface import SygnetModel

from random import choice, choices
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

def tune(
    parameter_dict,
    data,
    perf_fun, 
    runs,
    model_opts = {},
    fit_opts = {},
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

    if mode != "wgan":
        return None

    if type(parameter_dict) is not dict:
        logger.error("`parameter_dict` must be a dictionary with hyperparameters as keys and lists of options to try as values. \n \
            Tunable hyperparameters are currently `layers`,`nodes`,`dropout_p`, `layer_norms`,`relu_leak`,`batch_size,`learning_rate`, and `adam_betas`")

    model_hyps = ['dropout_p','relu_leak','layer_norms', 'hidden_layers']
    fit_hyps = ['batch_size','learning_rate','adam_betas']

    model_dict = dict((k, parameter_dict[k]) for k in model_hyps if k in parameter_dict)
    fit_dict = dict((k, parameter_dict[k]) for k in fit_hyps if k in parameter_dict)

    tuning_results = []

    for i in range(runs):

        if parameter_dict.get('layers') is not None and parameter_dict.get('nodes') is not None:
            model_dict['hidden_layers'] = choices(parameter_dict.get('nodes'), k = parameter_dict.get('layers'))

        kf = KFold(n_splits=5)
        kf.get_n_splits(data)

        for train_idx, kth_idx in kf.split(data):

            sygnet_model = SygnetModel(**model_dict, **model_opts)

            sygnet_model.fit(
                data.iloc[train_idx,:],
                **fit_dict,
                **fit_opts)

            synth_data = sygnet_model.sample(n)

            k_out = perf_fun(data = synth_data)

            tuning_results.append([i, kth_idx, k_out])
        
        tuning_results = pd.DataFrame(tuning_results)
        tuning_results.columns = ["it", "k-fold", "fun_out"]
