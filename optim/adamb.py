from collections import defaultdict
from typing import Callable, Tuple, Dict

import torch
import math
from torch import Tensor, nn
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.optim.optimizer import Optimizer


__all__ = ['AdamB', 'adamb']


### Note that the list of normalizations is taken from pytorch v1.9.0 ###
### https://pytorch.org/docs/1.9.0/nn.html#normalization-layers ###

NORMALIZATION_LAYERS = [
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LazyBatchNorm1d,
    nn.LazyBatchNorm2d,
    nn.LazyBatchNorm3d,
    nn.GroupNorm,
    nn.SyncBatchNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.LocalResponseNorm, # but no training params
]


class AdamB(Optimizer):
    """Implementation of AdamB optimizer

    Attributes:
        params:
        lr:
        betas:
        eps:
        init_std_scales:
        reg_factor:
    """

    def __init__(
        self,
        model: nn.Module,
        reg_factor: float = 1e-2,
        lr: float = 1e-3,
        lr_rho: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        init_std_scale: int = 1.0,
        device="cpu",
        log_s0: float = -1.0,
        log_s1: float = -6.0,        
        adam_eps: float = 1e-8,
        non_bayes_list: list = None,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_rho < 0.0:
            raise ValueError(f"Invalid learning rate: {lr_rho}")
        if not (0 <= betas[0] <= 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0 <= betas[1] <= 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if adam_eps < 0.0:
            raise ValueError(f"Invalid epsilon of adam: {adam_eps}")
        if reg_factor < 0.0:
            raise ValueError(f"Invalid number of decoupled regularization factor: {reg_factor}")

        self.model = model

        defaults = dict(
            reg_factor=reg_factor,
            lr=lr,
            lr_rho=lr_rho,
            b1=betas[0],
            b2=betas[1],
            adam_eps=adam_eps,
            init_std_scale=init_std_scale,
        )

        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []
        self.device = device
        self.s0 = torch.exp(Tensor([log_s0])).to(device)
        self.s1 = torch.exp(Tensor([log_s1])).to(device)

        for module in self.model.modules():
            if len(list(module.children())) > 0:
                continue

            params = list(module.parameters())

            if len(params) == 0:
                continue

            group = {
                "params": params,
                "non_bayes": False,
                "extracted_layer": str(module), # for debug 
            }

            ### AdamB is not used for normalizations, but Adam is used w/o any regularization ###
            group["non_bayes"] = isinstance(module, tuple(NORMALIZATION_LAYERS))

            # If the module is in the non_bayes_list, then it is not used AdamB
            if non_bayes_list is not None:
                for non_bayes in non_bayes_list:
                    group["non_bayes"] = group["non_bayes"] or isinstance(module, non_bayes)

            self.add_param_group(group)

        self.init_params()

    def __setstate__(self, state) -> None:
        super(AdamB, self).__setstate__(state)

    @torch.no_grad()
    def init_params(self) -> None:
        """Initializing all parameters in the network"""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                # Common states of module
                state["b1_decay"] = self.defaults["b1"]
                state["b2_decay"] = self.defaults["b2"]
                state["adam_m"] = torch.zeros(p.data.shape).to(self.device)
                state["adam_v"] = torch.zeros(p.data.shape).to(self.device)

                # For the modules which need sampling
                if group["non_bayes"]:
                    continue

                size = len(p.data.shape)
                xi = self.defaults["init_std_scale"]

                if size == 1:
                    n = p.data.shape[0]
                    xavier_std = math.sqrt(2.0 / n)
                elif size >= 2:
                    n_in, n_out = _calculate_fan_in_and_fan_out(p.data)
                    xavier_std = math.sqrt(2.0 / (n_in + n_out))


                state["mu"] = torch.normal(
                    mean=torch.zeros(p.data.shape), std=xavier_std
                ).to(self.device)

                state["rho"] = (
                    torch.ones(size=p.data.shape)
                    * math.log(math.exp(xi * xavier_std) - 1.0)
                ).to(self.device)

    def calc_log_posterior(self) -> Tensor:
        log_posterior = 0.0
        for group in self.param_groups:
            if group["non_bayes"]:
                continue

            for p in group["params"]:
                state = self.state[p]
                eps = state["epsilon"].to(self.device)
                rho = state["rho"].to(self.device)
                sigma = torch.log(1.0 + torch.exp(rho))
                log_prob = -torch.log(2 * math.pi * sigma ** 2) / 2 - eps ** 2 / (
                    2 * sigma ** 2
                )
                log_posterior += log_prob.sum()


        return log_posterior

    def calc_log_sm_prior(self) -> Tensor:
        """
        Considering Gaussian Scale Mixtures
        $$
        p(x|pi, m, s) = \sum_{i=0}^1 pi_i Normal(x|m_i, s_i)
        $$
        ,we assumed that $m_0 = 0, m_1 = 0 and pi_0 = a$.
        """
        log_prior = 0.0

        m0 = 0.0
        s0 = self.s0
        m1 = 0.0
        s1 = self.s1
        a = 0.5
        eps = 1e-8

        def normal(x, mean, sd):
            return torch.exp(-((x - mean) ** 2) / (2 * sd ** 2)) / torch.sqrt(2 * math.pi * sd)

        for group in self.param_groups:
            if group["non_bayes"]:
                continue

            for p in group["params"]:
                w = p.data
                log_prior += torch.log(
                    a * normal(w, m0, s0) + (1.0 - a) * normal(w, m1, s1) + eps
                ).sum()

        return log_prior

    def calc_derivative_of_log_sm_prior(self, x, s0=torch.exp(Tensor([-1.0])), s1=torch.exp(Tensor([-6.0]))):
        """
        Considering Gaussian Scale Mixtures
        $$
            p(x|pi, m, s) = \sum_{i=0}^1 pi_i Normal(x|m_i, s_i)
        $$
        ,we assumed that $m_0 = 0, m_1 = 0 and pi_0 = a$.
        """

        m0 = 0.0
        m1 = 0.0
        a0 = 0.5
        a1 = 1.0 - a0

        tmp_term = (-1.0 / (s1 ** 2)) + (1.0 / (s0 ** 2))  # (-1/s_1^2 + 1/s_0^2)

        numerator = a1 * tmp_term * x * \
            (s0 / s1) * torch.exp(((x ** 2) / 2.0) * tmp_term)
        denominator = a0 + (a1 * (s0 / s1)) * \
            torch.exp(((x ** 2) / 2.0) * tmp_term)

        return -x / (s0 ** 2) + numerator / denominator

    def sample_params(self, sampling: bool = True, sample_conv: bool = True, sample_fc: bool = True) -> None:
        """Sampling all parameters in the network"""

        for group in self.param_groups:
            if group["non_bayes"]:
                continue

            for p in group["params"]:
                state = self.state[p]
                mu = state["mu"]
                rho = state["rho"]
                sigma = torch.log(1.0 + torch.exp(rho))

                if sampling:
                    #epsilon = torch.normal(
                    #    torch.zeros(sigma.shape), torch.ones(sigma.shape)
                    #).to(self.device)
                    zeros_cuda = torch.zeros_like(sigma)
                    ones_cuda = torch.ones_like(sigma)
                    epsilon = torch.normal(zeros_cuda, ones_cuda) # Sampling on GPU
                    w = mu + sigma * epsilon
                    p.data = w
                    state["epsilon"] = epsilon
                else:
                    p.data = mu
                    state["epsilon"] = torch.zeros_like(sigma) #torch.zeros(sigma.shape).to(self.device)


    def adam(
        self,
        m: Tensor,
        v: Tensor,
        grad: Tensor,
        b1: float,
        b2: float,
        b1_decay: float,
        b2_decay: float,
        adam_eps: float,
    ) -> Tensor:
        g = grad
        m.mul_(b1).add_(g, alpha = 1.0 - b1)
        v.mul_(b2).addcmul_(g, g, value = 1.0 - b2)
        m_hat = m / (1.0 - b1_decay)
        v_hat = v / (1.0 - b2_decay)
        adam_grad = m_hat / (torch.sqrt(v_hat).add_(adam_eps))
        return adam_grad

    def step(self, closure, sampling=True):
        return self._step(closure, sampling)

    def _step(
        self, closure: Callable[[], Tuple[Tensor, Tensor]], sampling=True
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Args:
            closure: The function includes the backward process and returns the tuple (loss, accuracy)

        Returns:
            This method returns the tuple (loss, accuracy) which calculates as sample average
        """

        b1 = self.defaults["b1"]
        b2 = self.defaults["b2"]
        adam_eps = self.defaults["adam_eps"]
        reg_factor = self.defaults["reg_factor"]

        loss = torch.Tensor([0.0]).to(self.device)
        acc = torch.Tensor([0.0]).to(self.device)

        self.sample_params(sampling)
        
        closure_outputs = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lr_rho = group["lr_rho"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = self.adam(
                    state["adam_m"],
                    state["adam_v"],
                    p.grad,
                    b1,
                    b2,
                    state["b1_decay"],
                    state["b2_decay"],
                    adam_eps,
                )
                if group["non_bayes"]:
                    p.data -= lr * grad
                else:
                    rho = state["rho"]
                    sigma = torch.log(1.0 + torch.exp(rho))
                    gp = self.calc_derivative_of_log_sm_prior(p.data, self.s0, self.s1)
                    grad_mu = grad - gp * reg_factor
                    if sampling:
                        grad_rho = (
                            grad_mu * state["epsilon"] - reg_factor * (1.0 / sigma)
                        ) / (1.0 + torch.exp(-rho))
                    else:
                        grad_rho = 0.0
                    state["mu"] -= lr * grad_mu
                    state["rho"] -= lr_rho * grad_rho
                state["b1_decay"] *= b1
                state["b2_decay"] *= b2
        return closure_outputs