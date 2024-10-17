r"""Rollout algorithms for forecasting"""

from abc import ABC, abstractmethod

import torch

import numpy as np
from pdediff.guidance import Tensor
from pdediff.mcs import *

from pdediff.score import *
from pdediff.guidance import *
from pdediff.mcs import Tensor
from pdediff.sampler.sampler import *
from pdediff.sampler.expint import *

from torch.func import vmap

#######################################################
##################    Rollout    ######################
#######################################################


def set_seed(seed):
    # force reproducibility
    torch.backends.cudnn.deterministic = True

    # set the seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class Rollout(ABC):
    # each sampler should have an unconditional score and a state space info
    # plus all the sampling parameters like predictive steps, correction steps, tau
    # that you get from **cfg.eval.sampling
    def __init__(
        self,
        unconditional_score: MCScoreNet,
        state_shape: tuple,
        steps: int,
        corrections: int,
        tau: float,
        sampler: Sampler = None,
    ):
        super().__init__()
        """
        Abstract sampler class.

        Args:
            unconditional_score: unconditional or prior score. Should be a MCScoreNet class.
            state_shape: Event shape, i.e. shape of a single trajectory state.
            steps: Number of diffusion steps, i.e. how many discretization steps we use in solving the reverse SDE.
            corrections: Number of Langevin corrections steps for each diffusion step.
            tau: parameters that approximates the SNR during the Langevin corrections. If corrections=0, this parameter has 0 influence.
            sampler: type of sampler used (Expint/DPM).
        """

        self.unconditional_score = unconditional_score
        self.state_shape = state_shape
        self.steps = steps
        self.corrections = corrections
        self.tau = tau
        self.sampler = sampler

    @abstractmethod
    def sample_traj(self, trajectory_length: int) -> Tensor:
        pass

    @abstractmethod
    def sample_traj_conditioning(self, *args, **kwargs) -> Tensor:
        pass

    def sample_traj(
        self, 
        trajectory_length, 
        y_conditioning,
        mask,
        seed=0, 
        batch_shape=(),
        *args,
        **kwargs,
    ):
        sampled_trajectories = []
        for y_cond_batch, mask_batch in zip(y_conditioning.split(batch_shape[0]), mask.split(batch_shape[0])):
            traj_batch = self.sample_traj_conditioning(
                trajectory_length,
                y_cond_batch,
                mask_batch,
                seed=seed,
                batch_shape=(y_cond_batch.shape[0],),
                *args,
                **kwargs,
            )
            sampled_trajectories.append(traj_batch)
        
        sampled_trajectories = torch.cat(sampled_trajectories)

        return sampled_trajectories


class ConditionalRolloutAllAtOnce(Rollout):
    def __init__(
        self,
        unconditional_score: MCScoreNet,
        state_shape: tuple,
        A,
        likelihoods: List[Likelihood],
        guidance: GuidedScore,
        likelihood_stds: List[float],
        steps: int,
        corrections: int,
        tau: float,
        gammas: float = None,
        sampler: Sampler = None,
        model_type: str = "noise",
    ):
        super().__init__(unconditional_score, state_shape, steps, corrections, tau, sampler)

        """
        Sampling all at once from the conditional score.

        Args:
            unconditional_score: unconditional or prior score. Should be a MCScoreNet class.
            state_shape: Event shape, i.e. shape of a single trajectory state.
            A: linear or non-linear mapping that from x gets the observatin y.
            likelihood: observation likelihood, i.e. p(y|x_0) = p(y|A(x)). Example can be Gaussian, i.e. p(y|x_0) = N(y|A(x), std^2) 
            guidance: guidance term, for now either SDA or DPS
            likelihood_std: standard deviation of the likelihood
            steps: Number of diffusion steps, i.e. how many discretization steps we use in solving the reverse SDE.
            corrections: Number of Langevin corrections steps for each diffusion step.
            tau: parameters that approximates the SNR during the Langevin corrections. If corrections=0, this parameter has 0 influence.
            gamma: approximation for the scaling the covariance of the guidance, when using SDA
            sampler: type of sampler used (Expint/DPM).
            model_type: the output of the network (noise/x_start/v_prediction).
        """

        self.A = A
        self.likelihoods = likelihoods
        self.guidance = guidance
        self.likelihood_stds = likelihood_stds

        assert len(likelihoods)==1, (
            f"For AAO rollout you should specify one likelihood for the conditioning " 
            f"on observations, but number of specified likelihoods is {len(likelihoods)}."
        )
        assert len(likelihood_stds)==1, (
            f"For AAO rollout you should specify one likelihood for the conditioning "
            f"on observations, but number of specified likelihood stds is {len(likelihood_stds)}."
        )
        assert len(gammas)==1, (
            f"For AAO rollout you should specify one likelihood for the conditioning "
            f"on observations, but number of specified gammas is {len(gammas)}."
        )
        self.gammas = gammas
        self.model_type=model_type

    def sample_traj_conditioning(
            self, 
            trajectory_length, 
            y_conditioning, 
            mask,
            seed, 
            batch_shape=(), 
        ):
        """
        Args:
           trajectory_length: length of the trajectory generated, i.e. number of states
           y_conditioning: observations we are conditioning on
           seed: seed used for reproducibility
           batch_shape: tuple stating how many trajectories we want to sample. For example: () or (1,) indicates one, (5,) indicates 5 and so on
        """
        set_seed(seed)

        likelihood = self.likelihoods[0](
            y=y_conditioning * mask, 
            A=self.A, 
            std=self.likelihood_stds[0], 
            mask=mask,
        )

        guided_score = self.guidance(
            VPSDE(self.unconditional_score,
                  model_type=self.model_type,
                  ),
            likelihoods=[likelihood],
            gammas=[self.gammas[0]],
        )

        # create reverse sde with guided score
        if torch.cuda.is_available():
            sde = VPSDE(
                guided_score, 
                shape=(trajectory_length, *self.state_shape),
                model_type=self.model_type,
            ).cuda()
        else:
            sde = VPSDE(
                guided_score, 
                shape=(trajectory_length, *self.state_shape),
                model_type=self.model_type,
            )

        sde.eval()

        if self.sampler is None:
            x = sde.sample(
                steps=self.steps,
                corrections=self.corrections,
                tau=self.tau,
                shape=batch_shape,
            ).cpu()
        else:
            x = self.sampler.sample(
                sde,
                shape=batch_shape,
            ).cpu()

        return x

class ConditionalARRollout(Rollout):
    def __init__(
        self,
        unconditional_score: MCScoreNet,
        state_shape: tuple,
        A,
        likelihoods: List[Likelihood],
        guidance: GuidedScore,
        likelihood_stds: List[float],
        markov_blanket_window_size: int,
        steps: int,
        corrections: int,
        tau: float,
        gammas: float = None,
        sampler: Sampler = None,
        model_type: str = "noise",
        task: str = "forecast",
    ):
        super().__init__(unconditional_score, state_shape, steps, corrections, tau, sampler)

        """
        Sampling autoregressively with reconstruction guidance. At the first step we condition on the observation ys, but for all the other
        steps we are conditioning on the latest generated states.

        Args:
            unconditional_score: unconditional or prior score. Should be a MCScoreNet class.
            state_shape: Event shape, i.e. shape of a single trajectory state.
            A: linear or non-linear mapping that from x gets the observatin y.
            likelihoods: list of observation likelihoods, i.e. p(y|x_0) = p(y|A(x)). Example can be Gaussian, i.e. p(y|x_0) = N(y|A(x), std^2). Note this is the likelihood we use only at the first step (t=0).
            guidance: guidance term, for now either SDA or DPS. We need this because at the first step we generate unconditionally, but then we condition on the previously generated states.
            likelihood_stds: list of standard deviations of the likelihoods.
            markov_blanket_window_size: size of the markov blanket, i.e. 2*order + 1
            steps: Number of diffusion steps, i.e. how many discretization steps we use in solving the reverse SDE.
            corrections: Number of Langevin corrections steps for each diffusion step.
            tau: parameters that approximates the SNR during the Langevin corrections. If corrections=0, this parameter has 0 influence.
            gammas: approximation for the scaling the covariance of the guidance, when using SDA.
            sampler: type of sampler used (Expint/DPM).
            model_type: the output of the network (noise/x_start/v_prediction).
            task: task to solve (forecast or data_assimilation).
        """
        self.markov_blanket_window_size = markov_blanket_window_size
        self.A = A
        self.likelihoods = likelihoods
        self.guidance = guidance
        self.likelihood_stds = likelihood_stds

        assert len(likelihoods)==2, (
            f"For AR rollout you should specify two likelihoods - "
            f"one for the conditioning on observations, and one for the AR conditioning, "
            f"but number of specified likelihoods is {len(likelihoods)}."
        )
        assert len(likelihood_stds)==2, (
            f"For AR rollout you should only specify one likelihood std - "
            f"one for the conditioning on observations, and one for the AR conditioning, "
            f"but number of specified likelihood stds is {len(likelihood_stds)}."
        )
        assert len(gammas)==2, (
            f"For AR rollout you should only specify one guidance strength - "
            f"one for the conditioning on observations, and one for the AR conditioning, " 
            f"but number of specified gammas is {len(gammas)}."
        )
        
        self.gammas = gammas
        self.model_type = model_type
        self.task = task


    def sample_traj_conditioning(
        self,
        trajectory_length,
        y_conditioning,
        mask_conditioning,
        n_conditioned_frame,
        predictive_horizon,
        seed,
        batch_shape=(),
    ):
        """
        Function that generate a trajectory conditioned on some observation y. The function works as follows:
            First iter: generate (n_conditioned_frame + predictive_horizon) states conditioned on y_conditioning
            all the other iter: generate (n_conditioned_frame + predictive_horizon) states conditioned on previously n_conditioned_frame generated states

        Args:
            trajectory_length: length of the trajectory generated, i.e. number of states.
            y_conditioning: observations we are conditioning on to generate the first subtrajectory.
            n_conditioned_frame: how many states we are conditioning to generate predictive_horizon state each step.
            predictive_horizon: number of states generated at each step.
            seed: seed used for reproducibility
            batch_shape: tuple stating how many trajectories we want to sample. For example: () or (1,) indicates one, (5,) indicates 5 and so on
        """

        assert (
            n_conditioned_frame + predictive_horizon >= self.markov_blanket_window_size
        ), "We do not allow n_conditioned_frame + predictive_horizon < markov_blanket_window_size."

        # reproducibility
        set_seed(seed)

        stride = predictive_horizon
        idx_start = 0
        subtrajectory_length = n_conditioned_frame + predictive_horizon
        idx_end = subtrajectory_length

        def A_autoregressive_step(x):
            return x[:, :n_conditioned_frame, ...]

        # starting the generation
        while idx_start + n_conditioned_frame < trajectory_length:
            sub_y = y_conditioning[:, idx_start:idx_end, :]
            sub_mask = mask_conditioning[:, idx_start:idx_end, :].clone()

            # At first step we only condition on observations
            if idx_start == 0:
                cond_on_obs_likelihood = self.likelihoods[0](
                    y=sub_y * sub_mask, 
                    A=self.A, 
                    std=self.likelihood_stds[0],
                    mask=sub_mask,
                )

                guided_score = self.guidance(
                    VPSDE(self.unconditional_score,
                          model_type=self.model_type,
                          ),
                    likelihoods=[cond_on_obs_likelihood],
                    gammas=[self.gammas[0]],
                )

                if torch.cuda.is_available():
                    sde = VPSDE(
                        guided_score, 
                        shape=(subtrajectory_length, *self.state_shape),
                        model_type=self.model_type,
                    ).cuda()
                else:
                    sde = VPSDE(
                        guided_score, 
                        shape=(subtrajectory_length, *self.state_shape),
                        model_type=self.model_type,
                    )

                sde.eval()

                if self.sampler is None:
                    x = sde.sample(
                        steps=self.steps,
                        corrections=self.corrections,
                        tau=self.tau,
                        shape=batch_shape,
                    ).cpu()
                else:
                    x = self.sampler.sample(
                        sde,
                        shape=batch_shape,
                    ).cpu()

                # at the first step we keep all the generated states
                sampled_traj = x.clone()
                print("First sampled traj", sampled_traj.shape)
            
            # At the next steps we condition on observations and previously generated states
            else:
                sub_mask[:, :n_conditioned_frame, :] = 0
                x_star = A_autoregressive_step(sampled_traj[:, idx_start:idx_end, ...])

                cond_on_obs_likelihood = self.likelihoods[0](
                    y=sub_y * sub_mask, 
                    A=self.A,  
                    std=self.likelihood_stds[0],
                    mask=sub_mask,
                )

                cond_AR_likelihood = self.likelihoods[1](
                    y=x_star,
                    A=A_autoregressive_step, 
                    std=self.likelihood_stds[1],
                    mask=None, 
                )

                # We now create a guided score using the two likelihoods
                guided_score = self.guidance(
                    VPSDE(self.unconditional_score,
                          model_type=self.model_type,
                        ),
                    likelihoods=[cond_on_obs_likelihood, cond_AR_likelihood],
                    gammas=self.gammas,
                )

                if torch.cuda.is_available():
                    sde = VPSDE(
                        guided_score, 
                        shape=(subtrajectory_length, *self.state_shape),
                        model_type=self.model_type,
                    ).cuda()
                else:
                    sde = VPSDE(
                        guided_score, 
                        shape=(subtrajectory_length, *self.state_shape),
                        model_type=self.model_type,
                    )

                sde.eval()

                if self.sampler is None:
                    x = sde.sample(
                        steps=self.steps,
                        corrections=self.corrections,
                        tau=self.tau,
                        shape=batch_shape,
                    ).cpu()
                else:
                    x = self.sampler.sample(
                        sde,
                        shape=batch_shape,
                    ).cpu()


                # At the next step we either don't resample the overlapping states (forecast)
                # Or we resample them (data assimilation)
                if self.task=="forecast":
                    sampled_traj = torch.cat(
                        (
                            sampled_traj,
                            x[:, (subtrajectory_length - predictive_horizon) :, ...],
                        ),
                        dim=1,
                    )
                elif self.task=="data_assimilation":
                    sampled_traj = torch.cat(
                        (
                            sampled_traj[:, :-n_conditioned_frame, :], 
                            x,
                        ), 
                        dim=1,
                    )
                else:
                    raise ValueError(f"{self.task} is not supported")
                print("Sampled traj shape", sampled_traj.shape)

            idx_start += stride
            idx_end += stride

        sampled_traj = sampled_traj[:, :trajectory_length, ...]

        return sampled_traj


class AmortizedRollout(Rollout):
    """Sampling from the amortized model."""

    def __init__(
        self,
        score: MCScoreNet,
        state_shape: tuple,
        steps: int,
        corrections: int,
        tau: float,
        conditioned_frame: int,
        predictive_horizon: int,
        sampler: Optional[Sampler] = None,
        likelihood: Optional[Likelihood] = None, 
        guidance: Optional[GuidedScore] = None,
        likelihood_std: Optional[float] = None,
        gamma: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            unconditional_score=score,
            state_shape=state_shape,
            steps=steps,
            corrections=corrections,
            tau=tau,
            sampler=sampler,
        )

        self.conditioned_frame = conditioned_frame
        self.predictive_horizon = predictive_horizon
        self.window = conditioned_frame + predictive_horizon
        input_dim = self.window
        self.shape = (input_dim * state_shape[0], *state_shape[1:])
        self.state_shape = state_shape
        self.multiplier = state_shape[0]
        self.sampler = sampler
        self.likelihood = likelihood
        self.guidance = guidance
        self.likelihood_std = likelihood_std
        self.gamma = gamma

    def sample_traj(
        self, 
        trajectory_length: int, 
        seed: int, 
        batch_shape: tuple = (), 
        conditions: Optional[Tensor] = None, 
        obs: Optional[Tensor] = None, 
        obs_mask: Optional[Tensor] = None,
    ):
        """
        Method that creates the reverse SDE using the unconditinal score and sample a batch of trajectories.

        Args:
           trajectory_length: length of the trajectory generated, i.e. number of states
           seed: seed used for reproducibility
           batch_shape: tuple stating how many trajectories we want to sample.
                        For example: () or (1,) indicates one, (5,) indicates 5 and so on
           conditions: condition initial states
        """
        gen_sequence = self.sample_traj_fixed_initial_state(
            trajectory_length=trajectory_length,
            seed=seed,
            batch_shape=(len(conditions),),
            conditions=conditions,
            obs=obs, 
            obs_mask=obs_mask,
        )
        return gen_sequence.reshape((gen_sequence.shape[0], -1, *self.state_shape))
    
    def sample_traj_conditioning(self, *args, **kwargs):
        return self.sample_traj(*args, **kwargs)

    def sample_traj_fixed_initial_state(
        self, 
        trajectory_length: int, 
        seed: int, 
        batch_shape: tuple = (), 
        conditions: Tensor = None, 
        obs: Optional[Tensor] = None,
        obs_mask: Optional[Tensor] = None, 
    ):
        set_seed(seed)
        sde = VPSDE(self.unconditional_score.kernel, shape=self.shape)
        if obs is not None:

            obs = obs.reshape((obs.shape[0], -1, *self.state_shape[1:]))
            obs_mask = obs_mask.reshape((obs.shape[0], -1, *self.state_shape[1:]))
            
            def A(x, mask):
                return x*mask
            
            likelihood = self.likelihood(
                y=obs[:, :self.multiplier*self.window], 
                A=A,
                std=self.likelihood_std,
                mask=obs_mask[:, :self.multiplier*self.window] 
            )
            guided_score = self.guidance(
                sde,
                [likelihood],
                [self.gamma])
            sde = VPSDE(
                guided_score,
                shape=self.shape,
            )
            
        if torch.cuda.is_available():
            sde = sde.cuda()

        if self.sampler is None:
            sampling_function = lambda: sde.sample(
                steps=self.steps, corrections=self.corrections, tau=self.tau, shape=batch_shape
            )
        else:
            sampling_function = lambda: self.sampler.sample(sde, shape=batch_shape)

        generated_length = 0
        generated_sequence = None

        x_condition = conditions

        if x_condition is not None and torch.cuda.is_available():
            x_condition = x_condition.float().cuda()

        zero_mask_ph = torch.zeros_like(x_condition)
        if obs is not None:
            current_obs = obs[:, :self.multiplier*self.window]
            current_mask = obs_mask[:, :self.multiplier*self.window]

        while generated_length < trajectory_length:
            if obs is None:
                sde.net.set_condition(x_condition)
            else:
                sde.net.sde.net.set_condition(x_condition)
                likelihood.set_observation(current_obs, current_mask)
            x = sampling_function()
            if generated_sequence is not None:
                generated_sequence = torch.cat([generated_sequence, 
                                                x[:, -self.predictive_horizon*self.multiplier:]
                                                ], dim=1)
                generated_length += self.predictive_horizon
            else:
                generated_sequence = x
                generated_length += self.window

            x_condition = generated_sequence[:, -self.conditioned_frame*self.multiplier:]
            mask = torch.ones_like(x_condition)

            x_condition = torch.cat([x_condition,
                                     zero_mask_ph[:, : self.predictive_horizon*self.multiplier]],
                                     dim=1)
            mask = torch.cat([mask,
                              zero_mask_ph[:, : self.predictive_horizon*self.multiplier]],
                            dim=1)
            
            x_condition = torch.cat([mask, x_condition], dim=1)

            if obs is not None:
                current_obs = obs[:, self.multiplier*(generated_length - self.conditioned_frame): 
                                  self.multiplier*(generated_length + self.predictive_horizon)]
                current_mask = obs_mask[:, self.multiplier*(generated_length - self.conditioned_frame): 
                                        self.multiplier*(generated_length + self.predictive_horizon)]
        
        return generated_sequence[:, :trajectory_length*self.multiplier].cpu()