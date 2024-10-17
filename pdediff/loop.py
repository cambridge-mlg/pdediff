import numpy as np
import torch

from collections import defaultdict
from torch import Tensor
from tqdm import trange
from typing import *

from .sde import VPSDE

def get_category_dict(result_dict, categories, losses):
    # Iterate over each category and its corresponding value
    for category, loss in zip(categories, losses):
        # Check if the category is already a key in the dictionary
        if category.item() not in result_dict:
            # If not, add it with an empty list as the value
            result_dict[category.item()] = []
        # Append the value to the list associated with the category key
        result_dict[category.item()].append(loss.item())
    return dict(result_dict)

def loop(
    sde: VPSDE,
    trainset: Tensor,
    validset: Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int = 256,
    epoch_size: int = 4096,
    batch_size: int = 64,
    scheduler: float = "linear",
    device: str = "cpu",
    log_loss_per_level: bool = False,
    ema = None,
    data_spatial: int = 1,
    **absorb,
) -> Iterator:

    print("Model")
    print(sde)   

    amortized = absorb['amortized']
    if amortized:
        window = absorb['window']
        if 'fixed_horizon' in absorb:
            fixed_horizon = absorb['fixed_horizon']
            if fixed_horizon:
                num_observed = data_spatial * (window - absorb['predictive_horizon'])
        else: 
            fixed_horizon = False
       
    for epoch in (bar := trange(epochs, ncols=88)):
        losses_train = []
        losses_valid = []
        losses_train_dict = defaultdict(list)
        losses_valid_dict = defaultdict(list)

        sde.train()

        i = torch.randint(len(trainset['data']), (epoch_size,))
        train_data = trainset['data'][i].to(device).split(batch_size)

        for x in train_data:
            optimizer.zero_grad()

            if amortized:
                x_condition = x.clone()

                if not fixed_horizon:
                    num_observed = data_spatial*np.random.randint(low=0, high=window)
                mask = torch.zeros_like(x_condition)
                mask[:, :num_observed] = 1.0

                if absorb['noise_std'] > 0:
                    x_condition = torch.normal(x_condition, absorb['noise_std'])

                x_condition = torch.cat([mask, mask*x_condition], dim=1)

                sde.net.set_condition(x_condition)

            if log_loss_per_level:
                losses, time_cat = sde.loss(x, log_loss_per_level = log_loss_per_level)
                losses_train_dict = get_category_dict(losses_train_dict, time_cat, losses)
                l = losses.mean()
            else:
                l = sde.loss(x, log_loss_per_level = False)
            l.backward()
            optimizer.step()
            if ema is not None:
                ema.update(sde.parameters())

            if ema is not None:
                ema.update(sde.parameters())

            losses_train.append(l.detach())

        # Valid
        sde.eval()

        i = torch.randint(len(validset['data']), (epoch_size,))

        valid_data = validset['data'][i].to(device).split(batch_size)

        with torch.no_grad():
            for x in valid_data:
                if amortized:
                    x_condition = x.clone()
                    
                    if not fixed_horizon:
                        num_observed = data_spatial*np.random.randint(low=0, high=window)
                    mask = torch.zeros_like(x_condition)
                    mask[:, :num_observed] = 1.0

                    if absorb['noise_std'] > 0:
                        x_condition = torch.normal(x_condition, absorb['noise_std'])

                    x_condition = torch.cat([mask, mask*x_condition], dim=1)

                    sde.net.set_condition(x_condition)
                
                if log_loss_per_level:
                    losses, time_cat = sde.loss(x, log_loss_per_level = log_loss_per_level)
                    losses_valid_dict = get_category_dict(losses_valid_dict, time_cat, losses)
                    losses_valid.append(losses.mean())
                else:
                    losses_valid.append(sde.loss(x, log_loss_per_level = False))
                
        loss_train = torch.stack(losses_train).mean().item()
        loss_valid = torch.stack(losses_valid).mean().item()
        lr = optimizer.param_groups[0]["lr"]

        bar.set_postfix(lt=loss_train, lv=loss_valid, lr=lr)

        scheduler.step()

        if log_loss_per_level:
            losses_train_mean = []
            losses_valid_mean = []
            for key in range(11):
                if key in losses_train_dict.keys():
                    losses_train_mean.append(np.mean(np.array(losses_train_dict[key])))
                else:
                    losses_train_mean.append(0)
                if key in losses_valid_dict.keys():
                    losses_valid_mean.append(np.mean(np.array(losses_valid_dict[key])))
                else:
                    losses_valid_mean.append(0)
            yield loss_train, loss_valid, lr, losses_train_mean, losses_valid_mean, epoch
        else:
            yield loss_train, loss_valid, lr, epoch