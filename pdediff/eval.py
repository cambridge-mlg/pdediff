r"""Evaluation metrics"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from pdediff.viz.plotting import (
    plot_1d_trajectories,
    plot_mean_and_std,
    plot_kolmogorov_vorticity_trajectories
)
from pathlib import Path

def get_training_dir(current_dir, cfg):
    """returns path to associated training directory"""
    path = Path(current_dir)
    parent_dir = path.parent
    if parent_dir.name != cfg.name:  # if override_dirname is empty
        lines = parent_dir.name.split(",")
        new_lines = []
        for line in lines:
            # has_opening_bracket = '[' in line
            # has_closing_bracket = ']' in line
            splits = line.split('=')
            if len(splits) == 1: #NOTE: means it's due a list with comma
                new_lines[-1] += ',' + line
            else:
                if 'eval' not in splits[0]:
                    new_lines.append(line)
        training_dir = ','.join(new_lines)
        training_path = parent_dir.parent.joinpath(training_dir)
    else:
        training_path = parent_dir
    training_path = training_path.joinpath(str(cfg.seed))
    return training_path


def save_metric(metric, save_dir, cfg):
    assert save_dir is not None, "Please provide a saving directory"
    score_path = cfg.eval.load_model_name
    if cfg.eval.rollout_type == 'all_at_once':
        if cfg.eval.task == "forecast":
            name_file = f"/{score_path}_{cfg.eval.task}_{cfg.sampler.name}_{cfg.eval.rollout_type}_steps_{cfg.eval.sampling.steps}_traj_length_{cfg.eval.forecast.trajectory_length}_window_{cfg.window}_seed_{cfg.seed}_skip_{cfg.sampler.skip_type}_corrections_{cfg.eval.sampling.corrections}_gamma_{cfg.eval.guidance.gamma}_gamma1_{cfg.eval.guidance.gamma1}_std_{cfg.eval.guidance.std}_{cfg.epochs}.pt"
        elif cfg.eval.task == "data_assimilation":
            if cfg.eval.DA.online == "True":
                DA_type = "online"
            else:
                DA_type = "offline"
            name_file = f"/{score_path}_{cfg.eval.task}_{DA_type}_{cfg.sampler.name}_{cfg.eval.rollout_type}_steps_{cfg.eval.sampling.steps}_traj_length_{cfg.eval.forecast.trajectory_length}_window_{cfg.window}_seed_{cfg.seed}_skip_{cfg.sampler.skip_type}_corrections_{cfg.eval.sampling.corrections}_gamma_{cfg.eval.guidance.gamma}_std_{cfg.eval.guidance.std}_perc_obs_{cfg.eval.DA.perc_obs}_init_{cfg.eval.DA.init_cond}_{cfg.eval.DA.sparsity}_{cfg.epochs}.pt"
        else:
            raise ValueError(f"Unsupported task {cfg.eval.task}; supported tasks are forecast and data_assimilation")
        torch.save(metric, save_dir + name_file)
    elif cfg.eval.rollout_type == 'autoregressive':
        if cfg.eval.task == "forecast":
            name_file = f"/{score_path}_{cfg.eval.task}_{cfg.sampler.name}_{cfg.eval.rollout_type}_steps_{cfg.eval.sampling.steps}_traj_length_{cfg.eval.forecast.trajectory_length}_window_{cfg.window}_seed_{cfg.seed}_skip_{cfg.sampler.skip_type}_H_{cfg.eval.forecast.predictive_horizon}_C_{cfg.eval.forecast.conditioned_frame}_gamma_{cfg.eval.guidance.gamma}_gamma1_{cfg.eval.guidance.gamma1}_{cfg.epochs}.pt"
        else:
            if cfg.eval.DA.online:
                DA_type = "online"
            else:
                DA_type = "offline"
            name_file = f"/{score_path}_{cfg.eval.task}_{DA_type}_{cfg.sampler.name}_{cfg.eval.rollout_type}_steps_{cfg.eval.sampling.steps}_traj_length_{cfg.eval.forecast.trajectory_length}_window_{cfg.window}_seed_{cfg.seed}_skip_{cfg.sampler.skip_type}_corrections_{cfg.eval.sampling.corrections}_H_{cfg.eval.forecast.predictive_horizon}_C_{cfg.eval.forecast.conditioned_frame}_gamma_{cfg.eval.guidance.gamma}_std_{cfg.eval.guidance.std}_perc_obs_{cfg.eval.DA.perc_obs}_init_{cfg.eval.DA.init_cond}_sparsity_{cfg.eval.DA.sparsity}_{cfg.epochs}.pt"           
        torch.save(metric, save_dir + name_file)
    else:
        raise ValueError(f"Unsupported rollout_type {cfg.eval.rollout_type}, needs to be all_at_once or autoregressive")
    return


def pearson_correlation(sampled_trajectories: Tensor, true_trajectories: Tensor, reduce_batch: bool = False):
    """
    NOTE: they are running this at every generation step, so they track always how the correlation. Not understand why tbh
        evolves.

        So they do the following:

        corr_during_rollout= []
        from t in range(n_generation):
            sampled_traj = sample()
            reference_traj = true_traj[:, 0:t, state_shape] # here I am assuming we have a batch shape

            corr =pearson_correlation(sampled_traj, reference_traj)
            corr_during_rollout.append(corr)

    But is should work even if we are using the fully sampled trajectories.
    Taken from https://github.com/phlippe/pdearena/blob/main/pdearena/modules/loss.py line 39

    Args:
        sampled_trajectories: expected shape B, T, (event_shape) [batch, trajectory_length, event_shape]
        true_trajectories: expected shape B, T, (event_shape)
        reduce_batch: if True returns the average across the batch dimension
    """

    B = sampled_trajectories.size(0)
    T = sampled_trajectories.size(1)

    sampled_trajectories = sampled_trajectories.reshape(B, T, -1)
    true_trajectories = true_trajectories.reshape(B, T, -1)

    sampled_traj_mean = torch.mean(sampled_trajectories, dim=(2), keepdim=True)
    true_traj_mean = torch.mean(true_trajectories, dim=(2), keepdim=True)

    # Unbiased since we use unbiased estimates in covariance
    # TODO: understand what we should do here
    sampled_traj_std = torch.std(sampled_trajectories, dim=(2), unbiased=False)
    true_traj_std = torch.std(true_trajectories, dim=(2), unbiased=False)

    corr = torch.mean(
        (sampled_trajectories - sampled_traj_mean) * (true_trajectories - true_traj_mean),
        dim=2,
    ) / (sampled_traj_std * true_traj_std).clamp(
        min=torch.finfo(torch.float32).tiny
    )  # shape (B, T)

    if reduce_batch:
        corr = torch.mean(corr, dim=0)

    return corr


def compute_pearson_corr(sampled_x, true_x, cfg, log_plot = True, logger = None, save = True, save_dir = None):
    # Pearson correlation
    pearson_corr = pearson_correlation(
        true_x.reshape(
            cfg.eval.forecast.n_samples, 
            cfg.eval.forecast.trajectory_length, 
            *cfg.data.grid_size),
        sampled_x.squeeze(2).reshape(
            cfg.eval.forecast.n_samples, 
            cfg.eval.forecast.trajectory_length, 
            *cfg.data.grid_size),
        reduce_batch=False,
    )

    if log_plot:
        assert logger is not None, "Set logger" 
        
        fig = plot_mean_and_std(
            [pearson_corr],
            [f"corr_mean_std"],
        )
        logger.log_plot(
            f"pearson_correlation_{cfg.sampler.name}_{cfg.eval.rollout_type}",
            fig,
            step=cfg.eval.sampling.steps,
        )
    
    if save:
        save_metric(pearson_corr, save_dir, cfg)

    return pearson_corr


def mse_error(
    sampled_trajectories: Tensor,
    true_trajectories: Tensor,
    reduce_batch: bool = False,
    get_cumulative_mse: bool = True,
):
    """
    Method to compute the mse between the true trajectories and the samples. I am assume that both
    sampled_trajectories and true_trajectories have both a batch dimention, i.e. both have shapes
    [batch, trajectory_length, event_shape]

    Args:
       sampled_trajectories: expected shape B, T, (event_shape) [batch, trajectory_length, event_shape]
       true_trajectories: expected shape B, T, (event_shape)
       reduce_batch: if True returns the average across the batch dimension
    """
    B = sampled_trajectories.size(0)
    T = sampled_trajectories.size(1)

    sampled_trajectories = sampled_trajectories.reshape(B, T, -1)
    true_trajectories = true_trajectories.reshape(B, T, -1)

    # now I can compute the MSE between these
    squared_loss = (true_trajectories - sampled_trajectories) ** 2  # [batch, trajectory_length, event_shape] shape

    per_state_diff = squared_loss.sum(tuple(range(2, squared_loss.ndim)))  # [batch, trajectory_length] shape
    # now I can compute the cumulative error per state

    # here at each state we sum the error of the previous state
    # if False, we just compute the per state_diff
    mse = per_state_diff.cumsum(1) if get_cumulative_mse else per_state_diff  # [batch, trajectory_length] shape

    if reduce_batch:
        # I can average across the batch dimension
        return mse.mean(0)
    else:
        # return something that has shape [batch, trajectory_length]
        return mse


def compute_mse(sampled_x, true_x, cfg, log_plot=True, logger=None, save=True, save_dir=None):
    # MSE
    mse = mse_error(
        true_x.reshape(
            cfg.eval.forecast.n_samples, 
            cfg.eval.forecast.trajectory_length, 
            *cfg.data.grid_size),
        sampled_x.squeeze(2).reshape(
            cfg.eval.forecast.n_samples, 
            cfg.eval.forecast.trajectory_length, 
            *cfg.data.grid_size),
        reduce_batch=False,
        get_cumulative_mse=False,
    )

    if log_plot:
        assert logger is not None, "Set logger" 
        logger.log_metrics(
            {   
                f"rmsd_mean_{cfg.sampler.name}_{cfg.eval.rollout_type}": (mse.mean(axis = 1).sqrt().mean()).item(),
                f"rmsd_std_{cfg.sampler.name}_{cfg.eval.rollout_type}": (mse.mean(axis = 1).sqrt().std()).item(),
                f"rmsd_se_{cfg.sampler.name}_{cfg.eval.rollout_type}": (mse.mean(axis = 1).sqrt().std()/np.sqrt(cfg.eval.forecast.n_samples)).item(),
            }
        )

        fig = plot_mean_and_std(
            [mse],
            ["mse_mean_std"],
        )

        logger.log_plot(
            f"mse_{cfg.sampler.name}_{cfg.eval.rollout_type}",
            fig,
            step=cfg.eval.sampling.steps,
        )
    
    if save:
        save_metric(mse, save_dir, cfg)

    return mse


def plot_trajectories(true_x, logger, num_plot_samples, plot_name, cfg):
    # Only plot num_plot samples, otherwise the image is too big
    if "kolmogorov" in cfg.name:
        test_plot_to_log = plot_kolmogorov_vorticity_trajectories(true_x[: num_plot_samples])
    else:
        test_plot_to_log = plot_1d_trajectories(true_x[: num_plot_samples])
    logger.log_plot(
        plot_name,
        test_plot_to_log,
        step=cfg.eval.sampling.steps,
    )
    plt.close()
    return