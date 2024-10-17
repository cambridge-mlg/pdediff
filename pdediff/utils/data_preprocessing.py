import torch
import numpy as np
import einops
import pdb

def get_true_x(data, cfg):
    if cfg.name=="burgers":
        max_traj_length=101
    elif "KS" in cfg.name:
        max_traj_length=640
    elif "kolmogorov" in cfg.name:
        max_traj_length=64
    true_x = data["data"][:cfg.eval.forecast.n_samples, :min(max_traj_length, cfg.eval.forecast.trajectory_length)]
    if cfg.eval.forecast.trajectory_length > max_traj_length:
        true_x = torch.cat([true_x, torch.zeros(true_x.shape[0], cfg.eval.forecast.trajectory_length - max_traj_length, *true_x.shape[2:])], dim = 1)
    true_x = true_x.reshape(cfg.eval.forecast.n_samples, cfg.eval.forecast.trajectory_length, *cfg.data.state_shape)
    return true_x


def get_conditioning(x, cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # In this case, the A operator is just the observation
    def A(x):
        # The conditioning information are just the observations
        if cfg.eval.task in ["forecast", "data_assimilation"]:  
            return x
        else:
            raise ValueError(f"{cfg.eval.task} is not supported")
    to_append = cfg.eval.forecast.predictive_horizon - (cfg.eval.forecast.trajectory_length - cfg.window)%cfg.eval.forecast.predictive_horizon
    # Supported tasks are forecast and data_assimilation
    if cfg.eval.task == "forecast":
        mask = torch.cat(
            (
                torch.ones_like(x[:, :cfg.eval.forecast.conditioned_frame, ...]),
                torch.zeros_like(x[:, cfg.eval.forecast.conditioned_frame:, ...])
            ),
            dim=1
        )
    elif cfg.eval.task == "data_assimilation":
        mask = np.zeros(x.shape)
        time_idx = np.arange(0, cfg.eval.forecast.trajectory_length, 1)
        # Sparsity space means that we observe a certain percentage of pixels (perc_obs)
        # at each time index
        if cfg.eval.DA.sparsity=="space":
            if cfg.data.spatial == 1:
                # Init_cond all means we observe the full initial C states
                # If perc_obs = 0 this collapses to forecasting
                if cfg.eval.DA.init_cond == "all":
                    space_idx = [np.arange(0, mask.shape[2], 1) for _ in range(cfg.eval.forecast.conditioned_frame)]
                    space_idx += [np.random.choice(mask.shape[2], 
                                                   int(cfg.eval.DA.perc_obs * mask.shape[2]), 
                                                   replace = False) for _ in 
                                                   range(len(time_idx) - cfg.eval.forecast.conditioned_frame)]
                # Init_cond random means we observe only perc_obs of the initial C states
                elif cfg.eval.DA.init_cond == "random":
                    space_idx = [np.random.choice(mask.shape[2], 
                                                  int(cfg.eval.DA.perc_obs * mask.shape[2]), 
                                                  replace = False) for _ in range(len(time_idx))]
                else:
                    raise ValueError(f"{cfg.eval.DA.init_cond} not supported for initial conditioning")

                for i in range(len(space_idx)):
                    mask[np.ix_(np.arange(mask.shape[0]),[i], space_idx[i])] = 1
            elif cfg.data.spatial == 2:
                if cfg.eval.DA.init_cond == "all":
                    # Assume initial C states are fully observed
                    mask[:,0:cfg.eval.forecast.conditioned_frame,:,:] = 1

                    for i in range(len(time_idx)-cfg.eval.forecast.conditioned_frame):
                        # Create mask
                        ps = np.ones((64,64))*cfg.eval.DA.perc_obs
                        sample = np.random.binomial(1, p=ps)
                        mask[:,cfg.eval.forecast.conditioned_frame+i,:,:,:]=sample
                elif cfg.eval.DA.init_cond == "random":
                    # Assume everything is only partially observed
                    for i in range(len(time_idx)):
                        ps = np.ones((64,64))*cfg.eval.DA.perc_obs
                        sample = np.random.binomial(1, p=ps)
                        mask[:,i,:,:,:]=sample
                else:
                    raise ValueError(f"{cfg.eval.DA.init_cond} not supported for initial conditioning")

            mask = torch.from_numpy(mask)

        # Sparsity space-time means that we observe a certain percentage of pixels (perc_obs)
        # in the space-time space
        elif cfg.eval.DA.sparsity == "space-time":
            probs = torch.ones(x.shape[1:])*cfg.eval.DA.perc_obs
            mask = torch.bernoulli(probs)
            if cfg.eval.DA.init_cond == "all":
                mask[:cfg.eval.forecast.conditioned_frame, ...] = 1
            mask = einops.repeat(mask, 'm n l -> k m n l', k=x.shape[0])
        else:
            raise ValueError(f"{cfg.eval.DA.sparsity} not supported for type of sparsity of observations")
        
        # In AR rollouts the final window might extend the trajectory beyond 
        # the desired trajectory_length (because of reminders when dividing)
        # so we append 0s to the mask to be the same shape as generated trajs
        if cfg.eval.DA.online:
            to_append = 0
    else:
        raise ValueError(f"{cfg.eval.task} is not supported")
    
    # If we assume noise in the initial conditions
    if cfg.eval.init_noise:
        y_true = torch.normal(A(x), cfg.eval.guidance.std)
    else:
        y_true = A(x)
    if ((to_append!=0) and (cfg.eval.rollout_type=="autoregressive")):
            mask = torch.cat(
                (
                    mask, 
                    torch.zeros(mask.shape[0], to_append, *list(mask.shape[2:]))
                ), 
                dim = 1
            )
    if ((to_append!=0) and (cfg.eval.rollout_type=="autoregressive")):
        y_true = torch.cat(
            (
                y_true, 
                torch.zeros(y_true.shape[0], to_append, *list(y_true.shape[2:]))
            ), 
            dim = 1
        )
    return y_true, mask


def get_y_DA_online(forecast_length, step, y_true, mask, spatial):
    if forecast_length is None:
        pass
    else:
        y_true = y_true.unfold(dimension = 1, size = forecast_length, step = step).clone()
        y_true = y_true.movedim(-1, 2)
        y_true = y_true.movedim(0, 1)
        mask = mask.unfold(dimension = 1, size = forecast_length, step = step).clone()
        mask = mask.movedim(-1, 2)
        mask = mask.movedim(0, 1)

    mask[:, :, step:] = 0
    return y_true, mask


def append_data_ar(y_true, mask, spatial, predictive_horizon, window):
    trajectory_length = y_true.shape[-spatial - 2] + 1
    to_append = predictive_horizon - (trajectory_length - window)%predictive_horizon
    mask = append_zeros(mask, spatial, to_append)
    y_true = append_zeros(y_true, spatial, to_append)
    return y_true, mask

def append_data(y_true, mask, spatial, predictive_horizon, window):
    if spatial == 1:
        extra_channel = 0
    else:
        extra_channel = 1
    trajectory_length = y_true.shape[-spatial - 1 - extra_channel] + 1
    to_append = predictive_horizon - (trajectory_length - window)%predictive_horizon
    mask_addition = torch.zeros(*list(mask.shape[:-spatial - 1 - extra_channel]), 
                                to_append, 
                                *list(mask.shape[-spatial-extra_channel:]))
    mask = torch.cat((mask, mask_addition), dim = -spatial - 1 - extra_channel)
    y_true = torch.cat((y_true, mask_addition), dim = -spatial - 1 - extra_channel)
    return y_true, mask

def append_zeros(x, spatial, to_append):
    x_addition = torch.zeros(
                    *list(x.shape[:-spatial - 2]), 
                    to_append, 
                    *list(x.shape[-spatial-1:])
        ).to(x.device)
    x = torch.cat((x, x_addition), dim = -spatial - 2)
    return x


def get_space_conditioning(x, cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
   
    mask = np.zeros(x.shape)
    time_idx = np.arange(0, cfg.eval.forecast.trajectory_length, 1)

    for i in range(len(time_idx)):
        ps = np.ones((64,64))*cfg.eval.DA.perc_obs
        sample = np.random.binomial(1, p=ps)
        mask[:,i,:,:,:] = sample
        
    mask = torch.from_numpy(mask)

    to_append = cfg.eval.forecast.predictive_horizon - (cfg.eval.forecast.trajectory_length - cfg.window)%cfg.eval.forecast.predictive_horizon
    
    if cfg.eval.DA.online:
        to_append = 0
    
    if to_append != 0:
        mask = torch.cat((mask, torch.zeros(mask.shape[0], to_append, *list(mask.shape[2:]))), dim = 1)
        x = torch.cat((x, torch.zeros(x.shape[0], to_append, *list(x.shape[2:]))), dim = 1)

    initial_conditions_mask = mask[:, :cfg.window] * 0
    if cfg.eval.DA.init_cond == "all":
        initial_conditions_mask[:, :cfg.eval.forecast.conditioned_frame] = 1.0
    
    return x[:, :cfg.window].clone(), initial_conditions_mask, x, mask

def get_space_time_conditioning(x, cfg):

    mask = torch.ones_like(x)
    x_prob = cfg.eval.DA.perc_obs
    observations_mask = torch.bernoulli(mask * x_prob)
    initial_conditions_mask = mask[:, :cfg.window] * 0
    if cfg.eval.DA.init_cond == "all":
        initial_conditions_mask[:, :cfg.eval.forecast.conditioned_frame] = 1.0

    to_append = cfg.eval.forecast.predictive_horizon - (cfg.eval.forecast.trajectory_length - 
                                                        cfg.window)%cfg.eval.forecast.predictive_horizon
    if cfg.eval.DA.online:
            to_append = 0
    
    if to_append != 0:
        observations_mask = torch.cat((
            observations_mask, 
            torch.zeros(observations_mask.shape[0], to_append, *list(observations_mask.shape[2:]))
        ), dim = 1)
    
    if cfg.eval.guidance.std_init > 0:
        y_true_initial = torch.normal(x[:, :cfg.window], cfg.eval.guidance.std_init)
    else:
        y_true_initial = x[:, :cfg.window]
    
    if cfg.eval.guidance.std_da > 0:
        y_true_obs = torch.normal(x, cfg.eval.guidance.std_da)
    else:
        y_true_obs = x.clone()

    if to_append != 0:
        y_true_obs = torch.cat((
            y_true_obs, 
            torch.zeros(y_true_obs.shape[0], 
            to_append, 
            *list(y_true_obs.shape[2:]))
        ), dim = 1)
    return y_true_initial, initial_conditions_mask, y_true_obs, observations_mask




