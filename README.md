# Continuum Model

A continuum model for twisted bilayer **MoTe₂** and **WSe₂**, facilitating the construction, diagonalization, and visualization of band structures with configurable parameters.

<!-- ## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Jupyter Notebook](#running-the-jupyter-notebook)
  - [Examples](#examples)
- [Contact](#contact) -->

## Usage

To use the continuum model, simply open and run the provided Jupyter Notebook (continuum_model.ipynb). The notebook contains everything you need to get started and provides four examples for different configurations:
- **Example 1**: Full model of twisted bilayer MoTe₂.
- **Example 2**: Reduced model of twisted bilayer MoTe₂.
- **Example 3**: Full model of twisted bilayer WSe₂.
- **Example 4**: Reduced model of twisted bilayer WSe₂.
**Note:** Due to the structural convention difference, the Chern number calculated using these parameters in 2.88° and 3.48° tMoTe₂ is opposite to that in our paper. We will update the parameters soon.

## Getting Started
### Prerequisites

- **Python 3.6+**
- **Required Libraries**: Ensure the following libraries are installed:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `joblib`
  - `tqdm` (for optional progress bars)

You can install the required libraries using:
```bash
pip install numpy scipy matplotlib joblib tqdm
```

## Citation

If you use our code for your research, please consider citing our paper:
```bibtex
@misc{zhang_Universal_2024,
  title = {Universal Moiré-Model-Building Method without Fitting: Application to Twisted MoTe₂ and WSe₂,
  author = {Zhang, Yan and Pi, Hanqi and Liu, Jiaxuan and Miao, Wangqian and Qi, Ziyue and Regnault, Nicolas and Weng, Hongming and Dai, Xi and Bernevig, B. Andrei and Wu, Quansheng and Yu, Jiabin},
  year = {2024},
  month = nov,
  eprint = {2411.08108},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2411.08108},
  url = {https://arxiv.org/abs/2411.08108}
}

