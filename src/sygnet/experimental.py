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
  
  data['treated'] = np.random.choice([0, 1], size=size, p=[proportion, 1-proportion]) 

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
