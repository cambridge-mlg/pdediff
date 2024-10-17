import torch
import os
import hydra
import math

import pdediff.utils as sda_utils
import pdediff.eval as pdediff_eval
import pdediff.sampling as sampling

from omegaconf import OmegaConf
from hydra.utils import call, instantiate
from pathlib import Path

from pdediff.sde import VPSDE
from pdediff.loop import loop
from pdediff.utils.loggers import LoggerCollection
from pdediff.sampler.utils import get_sampler
from pdediff.score import make_score, load_score
from pdediff.nn.ema import ExponentialMovingAverage
from pdediff.utils.data_preprocessing import get_true_x, get_conditioning, get_space_time_conditioning, get_space_conditioning
import pdediff.rollout as rollout
import pdediff.guidance as guidance
import pdediff.viz.plotting as viz
import matplotlib.pyplot as plt
import numpy as np
from pdediff.mcs import curl
from pdediff.mcs import KolmogorovFlow


def check_experiment_name(name: str, amortized: bool = False):
    if amortized:
        return (
            name == 'KS_conditional_SDA' or 
            name == 'burgers_conditional_SDA' or 
            name == 'kolmogorov_conditional' or 
            name == 'kolmogorov_conditional_PDERef' or 
            name == 'kolmogorov_plain_amortized' or
            name == 'KS_conditional_PDERef'
            )
        
    return (name == 'KS_joint_SDA' or 
            name == 'burgers_joint' or 
            name == 'kolmogorov_joint_SDA' or 
            name =="KS_joint_PDERef" or 
            name == "kolmogorov_joint_PDERef"
            )


@hydra.main(config_path="config", config_name="main", version_base="1.3.2")
def main(cfg):
    os.environ["HYDRA_FULL_ERROR"] = "1"
    cfg_to_save = OmegaConf.to_container(cfg, resolve=True)

    current_dir = os.getcwd()
    ckpt_path = Path(os.path.join(current_dir, cfg.ckpt_dir, "score_last.pth"))
    ckpt_path_ema = Path(os.path.join(current_dir, cfg.ckpt_dir, "score_ema.pth"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path_latest = Path(os.path.join(current_dir, cfg.ckpt_dir, "latest.pth"))

    loggers = [instantiate(logger_config) for logger_config in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(cfg_to_save)

    window = cfg.window

    if cfg.amortized:
        condition_dim = window
    else:
        condition_dim = 0

    # Data
    print("Loading data")
    
    if cfg.mode in ["train", "all"]:
        trainset = sda_utils.load_data(os.path.join(cfg.data.path, "train.h5"), 
                            window=window, 
                            spatial=cfg.data.spatial)
        validset = sda_utils.load_data(os.path.join(cfg.data.path, "valid.h5"), 
                            window=window, 
                            spatial=cfg.data.spatial)
    if cfg.mode in ["eval", "all"]:
        test_dataset = sda_utils.load_dataset(os.path.join(cfg.data.path, "test.h5"))
        print("Test data min max",test_dataset['data'].min(), test_dataset['data'].max(), test_dataset['data'].shape)

    # load_dataset

    # Network
    print("Making the score")

    score = make_score(
        cfg.score,
        cfg.net,
        window,
        cfg.data.spatial,
        condition_dim=condition_dim*2,
    )

    shape = (window * cfg.data.spatial, *cfg.data.grid_size)

    sde = VPSDE(
        eps=score.kernel,
        shape=shape,
        model_type=cfg.model_type,
    ).cuda()

    ema = ExponentialMovingAverage(score.parameters(), decay=cfg.ema_decay)
    
    optimizer = instantiate(cfg.optim, params=sde.parameters())

    scheduler = cfg.scheduler_name
    epochs = cfg.epochs
    if scheduler == "linear":
        lr = lambda t: 1 - (t / epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
    elif scheduler == "cosine":
        # lr = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, 
            T_max=epochs, 
            eta_min=1e-6)
    elif scheduler == "exponential":
        lr = lambda t: math.exp(-7 * (t / epochs) ** 2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
    else:
        raise ValueError()
    
    if os.path.exists(ckpt_path_latest):
        latest_model = torch.load(ckpt_path_latest)
        sde.load_state_dict(latest_model['model'])
        ema.load_state_dict(latest_model['ema'])
        optimizer.load_state_dict(latest_model['optimizer'])
        cfg.epochs = cfg.epochs - latest_model['epoch']
        epoch_offset = latest_model['epoch']
        scheduler.load_state_dict(latest_model['scheduler']) 
    else: 
        epoch_offset = 0

    if cfg.mode in ["train", "all"]:
        generator = loop(
            sde=sde,
            trainset=trainset,
            validset=validset,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cuda",
            ema=ema,
            data_spatial=cfg.data.spatial,
            **cfg,
        )

        if cfg.log_loss_per_level:
            for i, (loss_train, loss_valid, lr, losses_train, losses_valid, epoch) in enumerate(generator):
                logger.log_metrics(
                    {
                        "loss_train": loss_train,
                        "loss_valid": loss_valid,
                        "lr": lr,
                        "loss_train0": losses_train[0],
                        "loss_train1": losses_train[1],
                        "loss_train2": losses_train[2],
                        "loss_train3": losses_train[3],
                        "loss_train4": losses_train[4],
                        "loss_train5": losses_train[5],
                        "loss_train6": losses_train[6],
                        "loss_train7": losses_train[7],
                        "loss_train8": losses_train[8],
                        "loss_train9": losses_train[9],
                        "loss_train10": losses_train[10],
                        "loss_valid0": losses_valid[0],
                        "loss_valid1": losses_valid[1],
                        "loss_valid2": losses_valid[2],
                        "loss_valid3": losses_valid[3],
                        "loss_valid4": losses_valid[4],
                        "loss_valid5": losses_valid[5],
                        "loss_valid6": losses_valid[6],
                        "loss_valid7": losses_valid[7],
                        "loss_valid8": losses_valid[8],
                        "loss_valid9": losses_valid[9],
                        "loss_valid10": losses_valid[10],
                    }
                )
                if (i + 1) % 100 == 0: 
                    checkpoint = {'epoch': i + 1 + epoch_offset,
                                'model': sde.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(), 
                                'ema': ema.state_dict()}
                    torch.save(checkpoint, ckpt_path_latest)              
        else:
            for i, (loss_train, loss_valid, lr, epoch) in enumerate(generator):
                logger.log_metrics(
                    {
                        "loss_train": loss_train,
                        "loss_valid": loss_valid,
                        "lr": lr,
                    }
                )
                if (i + 1) % 100 == 0: 
                    checkpoint = {'epoch': i + 1 + epoch_offset,
                                'model': sde.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(), 
                                'ema': ema.state_dict()}
                    torch.save(checkpoint, ckpt_path_latest)     
        # Save
        print(ckpt_path)
        torch.save(score.state_dict(), ckpt_path)
        with ema.average_parameters(score.parameters()):
            torch.save(score.state_dict(), ckpt_path_ema)

    if cfg.mode in ["eval", "all"]:
        if cfg.mode == "eval":
            print("Loading the ckpt")
            training_dir = pdediff_eval.get_training_dir(current_dir, cfg)
            ckpt_path = Path(os.path.join(training_dir, cfg.ckpt_dir, cfg.eval.load_model_name))
            score = load_score(ckpt_path)
            print('Checkpoint path', ckpt_path)

        sampler = get_sampler(cfg)

        if "conditional" not in cfg.name and 'amortized' not in cfg.name:

            if not check_experiment_name(cfg.name, False):
                raise NotImplementedError(f'{cfg.name} experiment is not implemented.')

            logger_dir = Path(os.path.join(current_dir, logger.log_dir, "sampling_data"))
            os.makedirs(logger_dir, exist_ok=True)

            if cfg.eval.conditioning == True:
                # Get the relevant entries from the test dataset
                true_x = get_true_x(test_dataset, cfg)
                test_batch_size = cfg.eval.forecast.test_batch_size

                # Plot the true test trajectories
                num_plot_samples = min(test_batch_size, 10)
                if "kolmogorov" not in cfg.name:
                    pdediff_eval.plot_trajectories(true_x, logger, num_plot_samples, plot_name="true_test_samples", cfg=cfg)

                # Get conditioning information
                y_true, mask = get_conditioning(true_x, cfg)

                if cfg.eval.task == "data_assimilation" and cfg.eval.DA.online:
                    sampled_x, true_x, true_mask = sampling.get_cond_DA_online(
                        score=score, 
                        y_true=y_true, 
                        sampler=sampler, 
                        cfg=cfg, 
                        logger=logger, 
                        mask=mask, 
                        true_x=true_x,
                    )
                    cfg.eval.forecast.n_samples = len(sampled_x)
                    cfg.eval.forecast.trajectory_length = sampled_x.shape[1]
                else:
                    if cfg.eval.rollout_type=="all_at_once":
                        sampled_x = sampling.get_cond_aao_samples(score, y_true, sampler, cfg, logger, mask)
                    elif cfg.eval.rollout_type=="autoregressive":
                        sampled_x = sampling.get_cond_ar_samples(score, y_true, sampler, cfg, logger, mask)
                    else:
                        raise ValueError(f"Unsupported rollout_type {cfg.eval.rollout_type}, needs to be all_at_once or autoregressive")
                    true_mask = None

                pdediff_eval.plot_trajectories(sampled_x[:5], logger, num_plot_samples, plot_name="samples", cfg=cfg)
                # pdediff_eval.plot_trajectories(true_x[:5], logger, num_plot_samples, plot_name="true_samples", cfg=cfg)

                # Save generated trajectory
                os.makedirs(os.path.join(logger_dir, "trajectories"), exist_ok=True)
                pdediff_eval.save_metric(sampled_x, os.path.join(logger_dir, "trajectories"), cfg)

                # Save mask
                if mask is not None:
                    os.makedirs(os.path.join(logger_dir, "masks"), exist_ok=True)
                    pdediff_eval.save_metric(mask, os.path.join(logger_dir, "masks"), cfg)

                if true_mask is not None:
                    os.makedirs(os.path.join(logger_dir, "true_masks"), exist_ok=True)
                    pdediff_eval.save_metric(true_mask, os.path.join(logger_dir, "true_masks"), cfg)

                # For Kolmogorov we compute the matrics on the vorticity, rather than velocity field
                if "kolmogorov" in cfg.name:
                    chain = KolmogorovFlow(size=256, dt=0.2)
                    # vorticity of test trajectories
                    true_x = chain.vorticity(true_x)
                    # vorticity of sampled trajectories
                    sampled_x = chain.vorticity(sampled_x)

                # Pearson correlation
                os.makedirs(os.path.join(logger_dir, "corr"), exist_ok=True)
                pearson_corr = pdediff_eval.compute_pearson_corr(sampled_x, 
                                     true_x, 
                                     cfg, 
                                     log_plot=True, 
                                     logger=logger, 
                                     save=True, 
                                     save_dir=os.path.join(logger_dir, "corr"),
                                     )

                os.makedirs(os.path.join(logger_dir, "mse"), exist_ok=True)
                mse = pdediff_eval.compute_mse(sampled_x, 
                            true_x, 
                            cfg, 
                            log_plot=True, 
                            logger=logger, 
                            save=True, 
                            save_dir=os.path.join(logger_dir, "mse")
                            )

                # Print metrics
                t = ((pearson_corr >= 0.8).sum(axis=1) * 0.2).mean()
                print('High correlation time = ', t)
                t = ((pearson_corr >= 0.8).sum(axis=1) * 0.2).std()
                print('High correlation time std = ', t)

                print('Mse = ', mse.mean(axis=1).sqrt().mean())
        
            elif cfg.eval.conditioning == False:
                sampled_x = sampling.get_uncond_aao_samples(score, sampler, cfg, logger)
            else:
                raise ValueError(f"Unsupported conditioning {cfg.eval.conditioning}")
        
        elif 'conditional' in cfg.name or 'amortized' in cfg.name:

            if not check_experiment_name(cfg.name, True):
                raise NotImplementedError(f'{cfg.name} experiment is not implemented.')
              
            if cfg.eval.task == 'data_assimilation':
                likelihood = guidance.Gaussian
                likelihood_std = cfg.eval.guidance.std
                gamma = cfg.eval.guidance.gamma
                if cfg.eval.guidance.type == "SDA":
                    guidance_type = guidance.SDA
                elif cfg.eval.guidance.type == "DPS":
                    guidance_type = guidance.DPS
                elif cfg.eval.guidance.type == 'VideoDiff':
                    guidance_type = guidance.VideoDiff
                elif cfg.eval.guidance.type == 'PGDM':
                    guidance_type = guidance.PGDM
                else:
                    raise ValueError(f"Guidance type is not supported")
            else:
                likelihood = None
                guidance_type = None
                likelihood_std = None
                gamma = None

            rollout_sampler = rollout.AmortizedRollout(
                score=score,
                state_shape=tuple(cfg.data.state_shape),
                sampler=sampler,
                conditioned_frame=cfg.eval.forecast.conditioned_frame, 
                predictive_horizon=cfg.eval.forecast.predictive_horizon,
                likelihood=likelihood,
                guidance=guidance_type, 
                likelihood_std=likelihood_std,
                gamma=gamma,
                **cfg.eval.sampling, 
            )

            print('Generating conditional samples')

            test_batch_size = cfg.eval.forecast.test_batch_size

            true_x = test_dataset['data'][:cfg.eval.forecast.n_samples, :cfg.eval.forecast.trajectory_length]

            print(f'True test samples size: {true_x.shape}')

            if "kolmogorov" in cfg.name:
                plot_fn = viz.plot_kolmogorov_vorticity_trajectories
            else:
                plot_fn = viz.plot_1d_trajectories

            test_plot_to_log = plot_fn(true_x[:12])
            logger.log_plot(
                "true_test_samples",
                test_plot_to_log,
                step=cfg.eval.sampling.steps,
            )
            plt.close()

            def prepare_initial_condition(x):
                batch_size = x.shape[0]
                conditioning = x[:, :cfg.window].reshape((batch_size, -1, *cfg.data.state_shape[1:]))
                mask = torch.zeros_like(conditioning)
                if cfg.eval.guidance.std_init > 0:
                    conditioning = torch.normal(conditioning, cfg.eval.guidance.std_init)
                mask[:, :cfg.eval.forecast.conditioned_frame*cfg.data.state_shape[0]] = 1.0
                return torch.cat([mask, mask*conditioning], dim=1)

            if cfg.eval.task == 'data_assimilation':
                if cfg.eval.DA.sparsity == "space-time":
                    initial_conditions, initial_conditions_mask, observations, observations_mask = get_space_time_conditioning(
                        true_x.clone(), 
                        cfg
                    )
                else:
                    initial_conditions, initial_conditions_mask, observations, observations_mask = get_space_conditioning(
                        true_x.clone(), 
                        cfg
                    )
                    
                    batch_size = initial_conditions.shape[0]
                    initial_conditions = initial_conditions.reshape((batch_size, -1, *cfg.data.state_shape[1:]))
                    initial_conditions_mask = initial_conditions_mask.reshape((batch_size, -1, *cfg.data.state_shape[1:]))
                        
                initial_conditions *= initial_conditions_mask
                initial_conditions = torch.cat([initial_conditions_mask, initial_conditions], dim=1)
                observations *= observations_mask
            else:
                initial_conditions = prepare_initial_condition(true_x)
                observations = None
                observations_mask = None

            if observations is not None:
                test_plot_to_log = plot_fn(observations[:12])
                logger.log_plot(
                    "observations",
                    test_plot_to_log,
                    step=cfg.eval.sampling.steps,
                )
                plt.close()
               
            if cfg.eval.task == "data_assimilation" and cfg.eval.DA.online:
                traj_autoreg, true_x = sampling.get_cond_DA_online_amortized(
                    obs=observations, 
                    sampler=rollout_sampler, 
                    cfg=cfg,
                    obs_mask=observations_mask,
                    initial_conds=initial_conditions,
                    true_x=true_x,
                )
                plot_to_log = plot_fn(traj_autoreg[:12])
                # let's start by considering conditional generation without residuals
                logger.log_plot(
                    f"amortized_conditional_samples_{cfg.sampler.name}",
                    plot_to_log,
                    step=cfg.eval.sampling.steps,
                )
                plt.close()
                cfg.eval.forecast.n_samples = len(traj_autoreg)
                trajectory_length = cfg.eval.DA.forecast_length 

            else:

                all_conditional_samples = []

                initial_conditions = initial_conditions.split(test_batch_size)

                if observations is None:
                    observations = [None]*len(initial_conditions)
                    observations_mask = [None]*len(initial_conditions)
                else:
                    observations = observations.split(test_batch_size)
                    observations_mask = observations_mask.split(test_batch_size)

                for batch_initial_conditions, obs, obs_mask in zip(initial_conditions, 
                                                                   observations, 
                                                                   observations_mask
                                                                  ):
                    conditional_samples = rollout_sampler.sample_traj(
                        cfg.eval.forecast.trajectory_length,
                        seed=cfg.seed,
                        batch_shape=(test_batch_size,),
                        conditions=batch_initial_conditions,
                        obs=obs, 
                        obs_mask=obs_mask
                    )
                    if len(all_conditional_samples) == 0:
                        plot_to_log = plot_fn(conditional_samples.squeeze(2)[:12])
                        logger.log_plot(
                            f"amortized_conditional_samples_{cfg.sampler.name}",
                            plot_to_log,
                            step=cfg.eval.sampling.steps,
                        )
                        plt.close()
                    all_conditional_samples += [conditional_samples]

                all_conditional_samples = torch.cat(all_conditional_samples, dim=0)

                traj_autoreg = all_conditional_samples.squeeze(2)

                trajectory_length = cfg.eval.forecast.trajectory_length

            if "kolmogorov" in cfg.name:
                true_x = curl(true_x)
                traj_autoreg = curl(traj_autoreg)

            pearson_correlation_value = pdediff_eval.pearson_correlation(
                true_x.reshape(
                    len(true_x),
                    trajectory_length,
                    -1),
                traj_autoreg.reshape(
                    len(true_x),
                    trajectory_length,
                    -1),
                reduce_batch=False,
            )

            fig = viz.plot_mean_and_std(
                [pearson_correlation_value],
                ["pearson correlation"],
            )

            logger.log_plot(
                f"amortized_pearson_correlation",
                fig,
                step=cfg.eval.sampling.steps,
            )
            plt.close()

            mse_value = pdediff_eval.mse_error(
                true_x.reshape(len(true_x),
                                trajectory_length,
                                -1),
                traj_autoreg.reshape(
                    len(true_x),
                    trajectory_length,
                    -1
                ),
                reduce_batch=False,
                get_cumulative_mse=False,
            )

            logger.log_metrics(
                {
                    f"rmsd_mean_{cfg.sampler.name}": (
                        mse_value.mean(axis=1).sqrt().mean()).item(),
                    f"rmsd_std_{cfg.sampler.name}": (
                        mse_value.mean(axis=1).sqrt().std()).item(),
                    f"rmsd_se_{cfg.sampler.name}": (
                        mse_value.mean(axis=1).sqrt().std() / np.sqrt(cfg.eval.forecast.n_samples)).item(),
                }
            )

            fig = viz.plot_metric_different_samples_and_average(mse_value, ["mse"])
            logger.log_plot(
                f"amortized_mse",
                fig,
                step=cfg.eval.sampling.steps,
            )
            plt.close()

            logger.log_pt_results(pearson_correlation_value, f'pearson_correlation.pt')

            t = ((pearson_correlation_value >= 0.8).sum(axis=1) * 0.2).mean()
            print('high correlation time = ', t)
            t = ((pearson_correlation_value >= 0.8).sum(axis=1) * 0.2).std()
            print('high correlation time std= ', t)

            print('mse = ', mse_value.mean(axis=1).sqrt().mean())

            logger.log_pt_results(mse_value, f'mse.pt')

            logger.log_pt_results(traj_autoreg, f'trajectories.pt')

    logger.close()


if __name__ == "__main__":
    main()