# SyGNet<img src="sygnet.png" alt="SyGNet Mascot" align="right" width="20%" /></a>

## **Sy**nthetic data using **G**enerative Adversarial **Net**works

*Principal Investigator: Dr Thomas Robinson (thomas.robinson@durham.ac.uk)*

*Research team: Artem Nesterov, Maksim Zubok*

**sygnet** is a Python package for generating synthetic data within social science contexts. The **sygnet** algorithm uses cutting-edge advances in deep learning methods to learn the underlying relationships between variables in a dataset. Users can then generate brand-new, synthetic observations that mimic the real data.

### Installation
To install via pip, you can run the following command at the command line:
`pip install sygnet`

**sygnet** requires:
    
    numpy>=1.20
    torch>=1.11.0
    scikit-learn>=1.0
    pandas>=1.4
    datetime
    tqdm

### Example implementation

You can find a demonstration of **sygnet** under [examples/basic_example](examples/basic_example.ipynb).

### Current version: 0.0.3 (alpha release)

**Alpha release**: You should expect both functionality and pipelines to change (rapidly). Comments and bug reports are very welcome!

Changes:

* Fixes column ordering issue when using mixed activation layer
* Updates example


### Previous releases

**0.0.2**
* Fixes mixed activation bug where final layer wasn't sent to `device`
* Adds `SygnetModel.transform()` alias for `SygnetModel.sample()`

**0.0.1**
Our first release! This version has been lightly tested and the core functionality has been implemented. 