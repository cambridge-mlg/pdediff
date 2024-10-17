r"""Utils/Functions to compute rollouts"""
import torch
import time
import pdediff.rollout as rollout
import pdediff.guidance as guidance
import pdediff.eval as pdediff_eval
from pdediff.viz.plotting import plot_kolmogorov_vorticity_trajectories
from pdediff.utils.data_preprocessing import get_y_DA_online, append_data_ar, append_data

# Conditional sampling AAO
def get_cond_aao_samples(score, y_true, sampler, cfg, logger, mask=None,):
    num_plot_samples = min(y_true.shape[0], 10)
    test_batch_size = cfg.eval.forecast.test_batch_size
    all_conditional_samples = []

    if cfg.eval.task in ["forecast", "data_assimilation"]:
        start = time.time()

        for (batch_y_true, batch_mask) in zip(y_true.split(test_batch_size), 
                                                          mask.split(test_batch_size) 
                                                          ):
            # The conditioning variables are the (masked) trajectories
            def A(x, _mask):
                return x * _mask

            conditional_sampler_all_at_once = rollout.ConditionalRolloutAllAtOnce(
                unconditional_score=score,
                state_shape=tuple(cfg.data.state_shape),
                A=A,
                likelihoods=[guidance.Gaussian],
                guidance=guidance.SDA,
                likelihood_stds=[cfg.eval.guidance.std],
                steps=cfg.eval.sampling.steps,
                corrections=cfg.eval.sampling.corrections,
                tau=cfg.eval.sampling.tau,
                gammas=[cfg.eval.guidance.gamma],
                sampler=sampler,
            )

            sampled_x = conditional_sampler_all_at_once.sample_traj(
                trajectory_length=cfg.eval.forecast.trajectory_length,
                y_conditioning=batch_y_true,
                mask=batch_mask,
                seed=cfg.seed,
                batch_shape=(test_batch_size,),
            )

            all_conditional_samples += [sampled_x]

        end = time.time()
        print(f"Computation took: {end - start} seconds")
        all_conditional_samples = torch.cat(all_conditional_samples, dim=0)
    else:
        raise ValueError(f"Unsupported task {cfg.eval.task}; supported tasks are forecast and data_assimilation")

    # Plot the resulting samples
    if cfg.name == "burgers" or cfg.name=='KS':
        pdediff_eval.plot_trajectories(
            all_conditional_samples,
            logger,
            num_plot_samples,
            plot_name=f"conditional_samples_{cfg.sampler.name}_{cfg.eval.rollout_type}",
            cfg=cfg,
        )
    else:
        image_vorticity = plot_kolmogorov_vorticity_trajectories(all_conditional_samples)
        logger.log_plot(
            f"AAO_samples_{cfg.sampler.name}",
            image_vorticity,
            step=cfg.eval.sampling.steps,
        )
    return all_conditional_samples

def get_cond_ar_samples(score, y_true, sampler, cfg, logger, mask = None):
    test_batch_size = cfg.eval.forecast.test_batch_size
    num_plot_samples = min(test_batch_size, 10)
    all_conditional_samples = []

    # Type of guidance
    if cfg.eval.guidance.type == "SDA":
        guidance_term = guidance.SDA
    elif cfg.eval.guidance.type == "DPS":
        guidance_term = guidance.DPS
    elif cfg.eval.guidance.type == 'VideoDiff':
        guidance_term = guidance.VideoDiff
    elif cfg.eval.guidance.type == 'PGDM':
        guidance_term = guidance.PGDM
    else:
        raise ValueError(f"{guidance_term} is not supported")
    
    # The conditioning variables are the (masked) trajectories
    def A(x, _mask):
        return x * _mask

    # Supported tasks are forecast and data_assimilation
    if cfg.eval.task in ["forecast", "data_assimilation"]:
        for batch_y_true, batch_mask in zip(y_true.split(test_batch_size), mask.split(test_batch_size)):
            
            conditional_sampler_autoregressive = rollout.ConditionalARRollout(
                    unconditional_score=score,
                    state_shape=tuple(cfg.data.state_shape),
                    A=A,
                    likelihoods=[guidance.Gaussian, guidance.Gaussian],
                    guidance=guidance_term,
                    likelihood_stds=[cfg.eval.guidance.std, cfg.eval.guidance.std],
                    markov_blanket_window_size=cfg.window,
                    steps=cfg.eval.sampling.steps,
                    corrections=cfg.eval.sampling.corrections,
                    tau=cfg.eval.sampling.tau,
                    gammas=[cfg.eval.guidance.gamma, cfg.eval.guidance.gamma],
                    sampler = sampler,
                    model_type=cfg.model_type,
                    task=cfg.eval.task,
            )

            t0 = time.time()
            conditional_samples = conditional_sampler_autoregressive.sample_traj(
                trajectory_length=cfg.eval.forecast.trajectory_length,
                y_conditioning=batch_y_true,
                mask=batch_mask,
                n_conditioned_frame=cfg.eval.forecast.conditioned_frame,
                predictive_horizon=cfg.eval.forecast.predictive_horizon,
                seed=cfg.seed,
                batch_shape=(test_batch_size,),
            )
            torch.cuda.current_stream().synchronize()
            t1 = time.time()
            if len(all_conditional_samples) == 0 and cfg.name != "kolmogorov":
                pdediff_eval.plot_trajectories(conditional_samples.squeeze(2), 
                    logger, 
                    num_plot_samples, 
                    plot_name=f"conditional_samples_{cfg.sampler.name}_{cfg.eval.rollout_type}",
                    cfg=cfg,
                )

            print("Computation time", t1-t0)
                
            all_conditional_samples += [conditional_samples]

    all_conditional_samples = torch.cat(all_conditional_samples, dim=0)

    return all_conditional_samples

def get_cond_DA_online(score, y_true, sampler, cfg, logger = None, mask = None, true_x = None):
    test_batch_size = cfg.eval.forecast.test_batch_size
    sampled_trajectories = []

    y_true_subtrajs, mask_subtrajs = get_y_DA_online(
        cfg.eval.DA.forecast_length, 
        cfg.eval.DA.step, 
        y_true, 
        mask, 
        cfg.data.spatial
    )

    true_x, true_mask = get_y_DA_online(
        cfg.eval.DA.forecast_length, 
        cfg.eval.DA.step, 
        true_x, 
        mask, 
        cfg.data.spatial
    )

    if cfg.eval.rollout_type == "autoregressive":
        y_true_subtrajs, mask_subtrajs = append_data_ar(
            y_true=y_true_subtrajs, 
            mask=mask_subtrajs, 
            spatial=cfg.data.spatial,
            predictive_horizon=cfg.eval.forecast.predictive_horizon, 
            window=cfg.window,
        )
    previous_state = torch.zeros_like(y_true_subtrajs[0, :, :1])
    previous_mask = torch.zeros_like(previous_state)
    for y_true_subtraj, mask_subtraj in zip(y_true_subtrajs, mask_subtrajs):
        y_true_subtraj = torch.cat(
            (previous_state, y_true_subtraj), 
            dim = 1)
        mask_subtraj = torch.cat(
            (previous_mask, mask_subtraj), 
            dim = 1)
        all_conditional_samples = []
        for batch_y_true, batch_mask in zip(y_true_subtraj.split(test_batch_size), 
                                                        mask_subtraj.split(test_batch_size) 
                                                        ):
            def A(x, _mask):
                return x * _mask
            
            if cfg.eval.rollout_type == "autoregressive":
                conditional_sampler_autoregressive = rollout.ConditionalARRollout(
                        unconditional_score=score,
                        state_shape=tuple(cfg.data.state_shape),
                        A=A,
                        likelihoods=[guidance.Gaussian, guidance.Gaussian],
                        guidance=guidance.SDA,
                        likelihood_stds=[cfg.eval.guidance.std, cfg.eval.guidance.std],
                        markov_blanket_window_size=cfg.window,
                        steps=cfg.eval.sampling.steps,
                        corrections=cfg.eval.sampling.corrections,
                        tau=cfg.eval.sampling.tau,
                        gammas=[cfg.eval.guidance.gamma, cfg.eval.guidance.gamma],
                        sampler = sampler,
                        model_type=cfg.model_type,
                        task=cfg.eval.task,
                )

                conditional_samples = conditional_sampler_autoregressive.sample_traj(
                    trajectory_length=cfg.eval.DA.forecast_length + 1,
                    y_conditioning=batch_y_true,
                    mask=batch_mask,
                    n_conditioned_frame=cfg.eval.forecast.conditioned_frame,
                    predictive_horizon=cfg.eval.forecast.predictive_horizon,
                    seed=cfg.seed,
                    batch_shape=(test_batch_size,),
                )

            elif cfg.eval.rollout_type == "all_at_once":
                conditional_sampler_all_at_once = rollout.ConditionalRolloutAllAtOnce(
                    unconditional_score=score,
                    state_shape=tuple(cfg.data.state_shape),
                    A=A,
                    likelihoods=[guidance.Gaussian],
                    guidance=guidance.SDA,
                    likelihood_stds=[cfg.eval.guidance.std],
                    steps=cfg.eval.sampling.steps,
                    corrections=cfg.eval.sampling.corrections,
                    tau=cfg.eval.sampling.tau,
                    gammas=[cfg.eval.guidance.gamma],
                    sampler=sampler,
                )

                conditional_samples = conditional_sampler_all_at_once.sample_traj(
                    trajectory_length=cfg.eval.DA.forecast_length + 1,
                    y_conditioning=batch_y_true,
                    mask=batch_mask,
                    seed=cfg.seed,
                    batch_shape=(test_batch_size,),
                )
            else:
                raise ValueError(f"Requested rollout type is not implemented.")

            conditional_samples = conditional_samples[:, 1:] # dropping first state that was used as initial state

            all_conditional_samples += [conditional_samples]
        
        all_conditional_samples = torch.cat(all_conditional_samples, dim=0)
        sampled_trajectories += [all_conditional_samples]

        previous_state = all_conditional_samples[:, cfg.eval.DA.step - 1].unsqueeze(1)
        previous_mask = torch.ones_like(previous_state)
    
    sampled_trajectories = torch.stack(sampled_trajectories, dim=0)
    return (
        sampled_trajectories.reshape(-1, *list(sampled_trajectories.shape)[2:]), 
        true_x.reshape(-1, *list(true_x.shape)[2:]), 
        true_mask.reshape(-1, *list(true_mask.shape)[2:])
    )

def get_cond_DA_online_amortized(
        obs, 
        obs_mask,
        sampler, 
        cfg, 
        initial_conds,
        true_x,
    ):
    test_batch_size = cfg.eval.forecast.test_batch_size
  
    obs_subtrajs, mask_subtrajs = get_y_DA_online(
        cfg.eval.DA.forecast_length, 
        cfg.eval.DA.step, 
        obs, 
        obs_mask, 
        cfg.data.spatial
    )
    
    true_x, _ = get_y_DA_online( cfg.eval.DA.forecast_length, 
        cfg.eval.DA.step, 
        true_x, 
        obs_mask, 
        cfg.data.spatial
    )

    # obs_subtrajs, mask_subtrajs, true_x = obs_subtrajs[:3], mask_subtrajs[:3], true_x[:3]
    obs_subtrajs, mask_subtrajs = append_data(
        y_true=obs_subtrajs, 
        mask=mask_subtrajs, 
        spatial=cfg.data.spatial,
        predictive_horizon=cfg.eval.forecast.predictive_horizon, 
        window=cfg.window,
    )

    current_conds = initial_conds
    previous_state = torch.zeros_like(obs_subtrajs[0, :, :1])
    previous_mask = torch.zeros_like(previous_state)

    zero_block = torch.repeat_interleave(previous_mask, repeats=cfg.window - 1, dim=1)

    sampled_trajectories = []

    for obs_subtraj, mask_subtraj in zip(obs_subtrajs, mask_subtrajs):
        obs_subtraj = torch.cat(
            (previous_state, obs_subtraj), 
            dim = 1)
        mask_subtraj = torch.cat(
            (previous_mask, mask_subtraj), 
            dim = 1)
        all_conditional_samples = []
        for batch_obs, batch_mask, batch_initial_cond in zip(
            obs_subtraj.split(test_batch_size), 
            mask_subtraj.split(test_batch_size), 
            current_conds.split(test_batch_size)
        ):
            conditional_samples = sampler.sample_traj(
                cfg.eval.DA.forecast_length + 1,
                seed=cfg.seed,
                batch_shape=(test_batch_size,),
                conditions=batch_initial_cond, 
                obs=batch_obs, 
                obs_mask=batch_mask
            ).squeeze(dim=2)

            conditional_samples = conditional_samples[:, 1:] 

            all_conditional_samples += [conditional_samples]
        
        all_conditional_samples = torch.cat(all_conditional_samples, dim=0)
        sampled_trajectories += [all_conditional_samples]

        previous_state = all_conditional_samples[:, cfg.eval.DA.step - 1].unsqueeze(1)
        previous_mask = torch.ones_like(previous_state)

        current_conds = torch.cat([previous_mask, zero_block, previous_state, zero_block], dim=1)
        current_conds = current_conds.reshape((current_conds.shape[0], -1, *cfg.data.state_shape[1:]))
    
    sampled_trajectories = torch.stack(sampled_trajectories, dim=0)
    return (
        sampled_trajectories.reshape(-1, *list(sampled_trajectories.shape)[2:]), 
        true_x.reshape(-1, *list(true_x.shape)[2:]), 
    )