# On conditional diffusion models for PDE simulations

This folder contains the code for the paper 'On conditional diffusion models for PDE simulations' accepted at NeurIPS 2024.  


## How to install

To use the code and reproduce results, you first need to create the environment and install all the dependencies by running the following commands

```
# create the environment
conda env create -f environment.yml
# activate the environment
conda activate pdediff
```

Once the environment is activated you can install the package

```
pip install -e .
```

## Code structure and organization
The main folder is `pdediff` which contains the following folders:

- `nn/`: files for implementing different U-Net architectures and related utils
- `sampler/`: files that implement different samplers for sampling from a diffusion model
- `utils/`: helper files for data preprocessing, logging and data loading
- `viz/`: files that contain functions used for plotting

and the following files:

- `eval.py`: functions used to evaluate the samples (e.g. pearson correlation
coeffiecient, mean squared error)
- `guidance.py`: contains the observation likelihood and the different reconstruction guidance strategies
- `loop.py`: functions used to train a diffusion model
- `mcs.py`: contains code to create a KolmogorovFlow
- `rollout.py`: contains the rollout procedures (autoregressive and all-at-once) to sample from the joint and amortised models
- `sampling.py`: contains code to obtain conditional/unconditional samples
- `score.py`: contains wrappers for the score network
- `sde.py`: contains code to create a noise scheduler for the variance preserving (VP) SDE

## Experiments

> Note: The datasets are too heavy to be shared, but they can be generated using the instructions from the paper.

We used `hydra` to manage config files and hyperparameters. All the data, model, sampler, and experiment configs and hyperparameters used to train the model can be found inside the `config` folder. 

The experiment configs usually have the following structure `{dataset}_{model_type}_{architecture}`. 
- The considered datasets are `burgers`, `KS`, and `kolmogorov`.
- The model_type can be `joint` or `conditional`. We also include configs for the `plain_amortized` models.
- The architecture can be `SDA` or `PDERef` (For burgers we only explore the `SDA` architecture).

We will now explain how to train a diffusion model on the KS dataset.

### Training a joint model on KS
If you want to train a model with the SDA architecture with an order $k$ assumption, i.e. pseudo Markov Blanket of size $2k+1$ you can run the following command

```
python main.py -m seed=0 mode=train experiment=KS_joint_SDA window={window_size}
```
where `window_size=2k+1`.


### Training a plain amortised diffusion model on KS

```
python main.py -m seed=0 mode=train experiment=KS_plain_amortized window={window_size} predictive_horizon={predictive_hor}
```

where `window_size` indicates the size of the Markov blanket, and `predictive_hor` indicates the number of frames that do not contain any conditioning information (H from the paper).

### Training a universal amortised diffusion model on KS

```
python main.py -m seed=0 mode=train experiment=KS_conditional_SDA window={window_size}
```

where `window_size` indicates the size of the Markov blanket.


### AR Conditional sampling on KS
If you want to sample from the trained joint model

```
python main.py -m seed=0 mode=eval experiment=KS_joint_SDA window={window_size} eval.forecast.conditioned_frame={n_conditioning} eval.forecast.predictive_horizon={predictive_hor} eval.sampling.corrections={n_corrections} eval.rollout_type="autoregressive" eval.task={task}
```

where `window_size` is the Markov blanket size, `n_conditioning` indicates the number of previous states we are conditioning on and `predictive_hor` the number of states we want to predict. These should usually sum up to the Markov blanket size, so it suffices to specify one of `n_conditioning`/`predictive_hor`, and the other will be adjusted accordingly. The value of `n_corrections` refers to the number of Langevin correction steps.

There are two options for the task:
- `task=forecast` - this performs forecasting where the conditioning information is the first fully-observed `n_conditioning` initial states.
- `task=data_assimilation` - this performs data assimilation (DA). Two scenarios are supported: offline `eval.DA.online=False` and online `eval.DA.online=True`. The percentage of observed data can be set through `eval.DA.perc_obs=x`, where $x\in [0,1]$.


### AAO Conditional sampling on KS

To sample using the AAO strategy, follow the instructions above with the only change of `eval.rollout_type="all_at_once"`. The number of conditioned frames and predictive horizon are not applicable in this scenario.

### Conditional sampling on KS using the plain amortised model

```
python main.py -m seed=0 mode=eval experiment=KS_plain_amortized_SDA window={window_size}
```

### Conditional sampling on KS using the universal amortised model

```
python main.py -m seed=0 mode=eval experiment=KS_conditional_SDA window={window_size} eval.forecast.conditioned_frame={n_conditioning} eval.forecast.predictive_horizon={predictive_hor} eval.sampling.corrections={n_corrections} eval.task={task}
```

where the universal amortised is able to use the two tasks available: `forecast` and `data_assimilation`. As in the joint model, only one of `n_conditioning`/`predictive_hor` would usually be specified and the other adjusted accordingly.

### Acknowledgements

We built our repo on the [Score-based Data Assimilation repo](https://github.com/francois-rozet/sda) that is publicly available.  For the results of the baselines presented in the paper we refer to the implementations available in the [PDERefiner repo](https://github.com/pdearena/pdearena). 