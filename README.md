# SyGNet<img src="sygnet.png" alt="SyGNet Mascot" align="right" height="139" /></a>

## **Sy**nthetic data using **G**enerative Adversarial **Net**works

*Principal Investigator: Dr Thomas Robinson (thomas.robinson@durham.ac.uk)*
*Research team: Artem Nesterov, Maksim Zubok*

**sygnet** is a Python package for generating synthetic data within social science contexts. The **sygnet** algorithm uses cutting-edge advances in deep learning methods to learn the underlying relationships between variables in a dataset, and to subsequently generate new observations that follow these observations.

### Installation
To install via pip, you can run the following command at the command line:
`pip install sygnet`

**sygnet** requires:
    
    * numpy>=1.20
    * torch>=1.11.0
    * scikit-learn>=1.0
    * pandas>=1.4
    * datetime
    * tqdm

### Version 0.0.1 (Alpha release)

Our first release -- this version has been lightly tested and the core functionality has been implemented. You should expect both functionality and architecture to change considerably. Comments and bug reports are very welcome!