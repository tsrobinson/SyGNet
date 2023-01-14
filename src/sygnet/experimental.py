from .requirements import *

def add_treatment(data,
                  outcome,
                  noise=False,
                  effect=None,
                  sd=None,
                  proportion=0.5,
                  seed=0):
  """Function to add treatment effect shocks

      Args:
          data (DataFrame): Synthetic data generated from model.sample
          outcome (str): Column name for a desired outcome variable from the dataframe
          effect (float): Teatment effect of a specified size
          noise (bool): True/false indicator denoting desired presence of noise 
          sd (float): Standard deviation for normal distribution governing effect and noise
          proportion (float): Proportion of treated observations
          seed (int): Random seed for replication

      Notes:
          Outcome must be a numeric variable

      Returns:
          Same dataframe with two added columns: treatment indicator and outcome column affected by treatment

  """
  
  np.random.seed(seed)

  size = len(data)
  
  data['treated'] = np.random.choice([0, 1], size=size, p=[1-proportion, proportion]) 

  # add treatment effect 
  if noise is True:
    # np.array with treatment effect for each observation
    effect_with_noise = np.random.normal(effect, sd, size)
    # add treatment effect (with noise) to treated observations
    data['treated_outcome'] = [x + e * t for x, e, t in zip(data[outcome], effect_with_noise, data['treated'])]
  else:
    # add treatment effect to treated observations
    data['treated_outcome'] = [x + effect * t for x, t in zip(data[outcome], data['treated'])]
  return data

def add_treatment_with_interaction(data,
                  outcome,
                  proportion,
                  seed,
                  effect=None,
                  sd=None,
                  add_interaction=False,                
                  interaction_variable=None,
                  int_effect=None,
                  int_sd=None,
                  proportion=0.5,
                  seed=0):
  """Function to add treatment effect shocks

  Args:
      data (DataFrame): Synthetic data generated from model.sample
      outcome (str): Column name from the dataframe
      effect (float): Teatment effect of a specified size
      noise (bool): True/false indicator denoting desired presence of noise
      sd (float): Standard deviation for normal distribution governing effect and noise
      add_interaction (bool): True/False indicator denoting presence of an interaction variable
      interaction_variable (str): Column name from the dataframe
      int_effect (float): interaction effect of a specified size
      int_sd (float): Standard deviation for normal distriution governing the interaction effet and noise
      proportion (float): Proportion of treated observations
      seed (int): Random seed for replication
      

  Notes:
      Outcome and interaction variables must be numeric

  Returns:
      Same dataframe with two added columns: treatment indicator and outcome column affected by treatment

  """
  np.random.seed(seed)

  size = len(data)
  
  data['treated'] = np.random.choice([0, 1], size=size, p=[1-proportion, proportion]) 
  effect_list = [effect]*len(data)
  # add treatment effect 

  if add_interaction is True:
    interaction_effect = np.random.normal(int_effect, int_sd, size)
    effect_mu = [t * e + z + t * z * i for i, e, z, t in zip(data[interaction_variable],
                                                             effect_list,
                                                             interaction_effect,
                                                             data['treated'])]
    # np.array with main treatment effect and interaction effect for each observation
    effect_with_noise = np.random.normal(effect_mu, sd, size)
    # add treatment effect (with noise) to treated observations
    data['treated_outcome'] = [x + e * t for x, e, t in zip(data[outcome], effect_with_noise, data['treated'])]

  else:
    # np.array with treatment effect for each observation
    effect_with_noise = np.random.normal(effect, sd, size)
    # add treatment effect (with noise) to treated observations
    data['treated_outcome'] = [x + e * t for x, e, t in zip(data[outcome], effect_with_noise, data['treated'])]

  return data


def add_heterogenous_treatment(data,
                  outcome,
                  effect=None,
                  sd=None,
                  add_heterogeneity=False,
                  control_heterogeneity=False,
                  third_variable=None,
                  amount_heterogeneity=1,
                  proportion=0.5,
                  seed=0):
  """Function to add treatment effect shocks

  Args:
      data (DataFrame): Synthetic data generated from model.sample
      outcome (str): Column name from the dataframe
      effect (float): Teatment effect of a specified size
      noise (bool): True/false indicator denoting desired presence of noise
      sd (float): Standard deviation for normal distribution governing effect and noise
      add_heterogeneity (bool): True/false indicator denoting the presence of effect heterogeneity
      control_heterogeneity (bool): True/false indicator denoting manual control of heterogeneity amonut
      third_variable (str): Column name from the dataframe
      amount_heterogeneity (float): the desired amount of heterogeneity
      proportion (float): Proportion of treated observations
      seed (int): Random seed for replication

  Notes:
      Outcome must be a numeric variable

  Returns:
      Same dataframe with two added columns: treatment indicator and outcome column affected by treatment

  """
  np.random.seed(seed)

  size = len(data)
  
  data['treated'] = np.random.choice([0, 1], size=size, p=[1-proportion, proportion]) 
  effect_list = [effect]*len(data)
  # add treatment effect 

  

  if add_heterogeneity is True:
    if control_heterogeneity is True:
      amount_list = [amount_heterogeneity]*len(data)
      var_z_score = [a*(z-np.mean(data[third_variable]))/np.std(data[third_variable]) for a, z in zip(amount_list, data[third_variable])]
      # np.array with main treatment effect and interaction effect for each observation
      effect_mu = [t * e * (1 + z) for t,e,z in zip(data['treated'], effect_list, var_z_score)]
      effect_with_noise = np.random.normal(effect_mu, sd, size)
      # add treatment effect (with noise) to treated observations
      data['treated_outcome'] = [x + e * t for x, e, t in zip(data[outcome], effect_with_noise, data['treated'])]

    else:
      var_z_score = [(z-np.mean(data[third_variable]))/np.std(data[third_variable]) for z in data[third_variable]]
      # np.array with main treatment effect and interaction effect for each observation
      effect_mu = [t * e * (1 + z) for t,e,z in zip(data['treated'], effect_list, var_z_score)]
      effect_with_noise = np.random.normal(effect_mu, sd, size)
      # add treatment effect (with noise) to treated observations
      data['treated_outcome'] = [x + e * t for x, e, t in zip(data[outcome], effect_with_noise, data['treated'])]


  else:
    # np.array with treatment effect for each observation
    effect_with_noise = np.random.normal(effect, sd, size)
    # add treatment effect (with noise) to treated observations
    data['treated_outcome'] = [x + e * t for x, e, t in zip(data[outcome], effect_with_noise, data['treated'])]

  return data
