from typing import Tuple, Union

import torch

from FunctionEncoder.Dataset.BaseDataset import BaseDataset


class Lorenz(BaseDataset):

    def __init__(self,
                 sigma_range: float = [0.1, 3.0],
                 rho_range: float = [0.1, 3.0],
                 beta_range: float = [0.1, 3.0],
                 dt: float = 0.05,
                 device: str = "auto",
                 dtype: torch.dtype = torch.float32,
                 n_functions:int=None,
                 n_examples:int=None,
                 n_queries:int=None,

                 # deprecated arguments
                 n_functions_per_sample:int = None,
                 n_examples_per_sample:int = None,
                 n_points_per_sample:int = None,

                 ):
        # default arguments. These default arguments will be placed in the constructor when the arguments are deprecated.
        # but for now they live here.
        if n_functions is None and n_functions_per_sample is None:
            n_functions = 10
        if n_examples is None and n_examples_per_sample is None:
            n_examples = 1000
        if n_queries is None and n_points_per_sample is None:
            n_queries = 10000

        super().__init__(input_size=(2,),
                         output_size=(2,),

                         data_type="deterministic",
                         device=device,
                         dtype=dtype,
                         n_functions=n_functions,
                         n_examples=n_examples,
                         n_queries=n_queries,

                         # deprecated arguments
                         total_n_functions=None,
                         total_n_samples_per_function=None,
                         n_functions_per_sample=n_functions_per_sample,
                         n_examples_per_sample=n_examples_per_sample,
                         n_points_per_sample=n_points_per_sample,

                         )

        self.input_min = torch.tensor([-4.0, -5.0], dtype=dtype, device=self.device)
        self.input_max = torch.tensor([4.0, 5.0], dtype=dtype, device=self.device)
        self.sigma_range = sigma_range
        self.rho_range = rho_range
        self.beta_range = beta_range
        self.dt = dt

    def sample(self) -> Tuple[  torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                dict]:
        with torch.no_grad():
            n_functions = self.n_functions
            n_examples = self.n_examples
            n_queries = self.n_queries

            # sample inputs
            sigmas = torch.rand(n_functions, 1, dtype=self.dtype, device=self.device) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
            rhos = torch.rand(n_functions, 1, dtype=self.dtype, device=self.device) * (self.rho_range[1] - self.rho_range[0]) + self.rho_range[0]
            betas = torch.rand(n_functions, 1, dtype=self.dtype, device=self.device) * (self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]
            example_xs = torch.rand(n_functions, n_examples, 2, dtype=self.dtype, device=self.device) * (self.input_max - self.input_min) + self.input_min
            query_xs = torch.rand(n_functions, n_queries, 2, dtype=self.dtype, device=self.device) * (self.input_max - self.input_min) + self.input_min

            # integrate to find outputs
            # this function only provides the change in state, not the state directly.
            example_ys = self.rk4_difference_only(example_xs, self.dt, sigmas, rhos, betas)
            query_ys = self.rk4_difference_only(query_xs, self.dt, sigmas, rhos, betas)

            return example_xs, example_ys, query_xs, query_ys, {"sigmas": sigmas, "rhos": rhos, "betas": betas, "dt": self.dt}

    def lorenz_dynamics(self, x, sigmas, rhos, betas):
        x1 = x[..., 0]
        x2 = x[..., 1]
        x3 = x[..., 2]
        dx1 = sigmas * (x2 - x1)
        dx2 = x1 * (rhos - x3) - x2
        dx3 = x1 * x2 - betas * x3
        return torch.stack([dx1, dx2, dx3], dim=-1)

    def rk4_difference_only(self, x0:torch.tensor, dt:torch.tensor,  sigmas:torch.tensor, rhos:torch.tensor, betas:torch.tensor) -> torch.tensor:
        k1 = self.lorenz_dynamics(x0, sigmas, rhos, betas)
        k2 = self.lorenz_dynamics(x0 + 0.5 * dt * k1, sigmas, rhos, betas)
        k3 = self.lorenz_dynamics(x0 + 0.5 * dt * k2, sigmas, rhos, betas)
        k4 = self.lorenz_dynamics(x0 + dt * k3, sigmas, rhos, betas)
        return (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
