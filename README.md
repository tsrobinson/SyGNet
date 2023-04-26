# SyGNet<img src="sygnet.png" alt="SyGNet Mascot" align="right" width="20%" /></a>

## **Sy**nthetic data using **G**enerative Adversarial **Net**works

*Principal Investigator: Dr Thomas Robinson (thomas.robinson@durham.ac.uk)*

*Research team: Artem Nesterov, Maksim Zubok*

![example workflow](https://github.com/tsrobinson/SyGNet/actions/workflows/python-app.yml/badge.svg)

**sygnet** ("sig·net") is a Python package for generating 
synthetic data within 
social science contexts. The **sygnet** algorithm uses cutting-edge advances in deep learning methods to learn the underlying relationships between variables in a dataset. Users can then generate brand-new, synthetic observations that mimic the real data.

### Installation
To install via pip, you can run the following command at the command line:
`pip install sygnet`

**sygnet** requires:
    
    numpy>=1.21
    torch>=1.10.0
    scikit-learn>=1.0
    pandas>=1.4
    datetime
    tqdm

### Example implementation

You can find a demonstration of **sygnet** under [examples/basic_example](examples/basic_example.ipynb).

### Current version: 0.0.9 (alpha release)

**Alpha release**: You should expect both functionality and pipelines to change (rapidly and without warning). Comments and bug reports are very welcome!

Changes:

* Rewrite of main interface and underlying functions
* Bulding models now structured in terms of hidden "blocks"
* Added self-attention mechanism

### Previous releases

**0.0.8**

Changes:

* Update `tune()` to provide no k-fold cross validation as default
* Update numpy dependency to fix pre-processing bug

**0.0.7**
* Update internal `train_*` functions to return losses and improve logging
* Update `tune()` function

**0.0.6 and 0.0.5**
* Internal changes to improve code efficiency
* Removes `sygnet_` from all submodule names
* Lowers PyTorch requirement to 1.10 for compatability with OpenCE environments

**0.0.4**
* Adds `tune()` function to run hyperparameter tuning
* Adds model saving functionality to `SygnetModel.fit()`
* Fixes various bugs
* Improves documentation

**0.0.3**
* Fixes column ordering issue when using mixed activation layer
* Updates example

**0.0.2**
* Fixes mixed activation bug where final layer wasn't sent to `device`
* Adds `SygnetModel.transform()` alias for `SygnetModel.sample()`

**0.0.1**
Our first release! This version has been lightly tested and the core functionality has been implemented. 
