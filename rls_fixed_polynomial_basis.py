
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.animation as animation

from FunctionEncoder import MultiDimQuadraticDataset, MSECallback, ListCallback, TensorboardCallback, DistanceCallback
from FunctionEncoder import FunctionEncoder
import argparse

from RLS.rls import *

"""
Tests the RLS algorithm on a polynomial model with a fixed polynomial basis.
"""

# seed torch
torch.manual_seed(0)

# create a dataset
n_output_dim = 1
a_range = (-3/50, 3/50)
b_range = (-3/50, 3/50)
c_range = (-3/50, 3/50)
input_range = (-10, 10)
dataset = MultiDimQuadraticDataset(a_range=a_range, b_range=b_range, c_range=c_range, input_range=input_range, n_output_dim=n_output_dim)

example_xs, example_ys, query_xs, query_ys, sample_info = dataset.sample()
function_idx = 3  # choose one of the functions
# print(f'Function {function_idx} A: {sample_info["As"][function_idx]}, B: {sample_info["Bs"][function_idx]}, C: {sample_info["Cs"][function_idx]}')
n = 100
example_xs = example_xs[function_idx, :n]
example_ys = example_ys[function_idx, :n]
query_xs   = query_xs[function_idx, :n]
query_ys   = query_ys[function_idx, :n]

# Make a polynomial model of degree 2
degree = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolynomialModel(
    input_size=(example_xs.shape[-1],),
    output_size=(example_ys.shape[-1],),
    degree=degree
)
theta0 = torch.zeros((degree+1), device=device)  # initial coefficients
rls = make_rls(method='standard', n_regressors=(degree+1), model=model, theta0=theta0)

plot_output_dim_num = 1 if n_output_dim > 1 else 0

parameter_iterates = [rls.theta.clone()]
for i in range(example_xs.shape[0]):
    x = example_xs[i,:].to(device)
    y = example_ys[i,:].to(device)
    theta, info = rls.update(x, y)
    parameter_iterates.append(theta.clone())

true_coefficients, sol_info = model.compute_representation(
    example_xs.to(device), example_ys.to(device)
)
true_coefficients = true_coefficients.view(-1)
rls_coefficients = parameter_iterates[-1].view(-1)
distance = torch.norm(rls_coefficients - true_coefficients, p=2)
print(f"Distance from iterative solution to the true solution: {distance.item()}")
print(f"True coefficients: {true_coefficients.cpu().numpy()}")
print(f"RLS coefficients: {rls_coefficients.cpu().numpy()}")
# print(f"Shape of the RLS coefficients: {rls_coefficients.shape}")
# print(f"Shape of the true coefficients: {true_coefficients.shape}")

# Subsample the coefficient iterates to reduce the number of frames if needed
subsample_factor = 1  # No subsampling if set to 1
subsampled_iterates = parameter_iterates[::subsample_factor]
query_xs, indicies = torch.sort(query_xs.squeeze(0), dim=-2)
query_ys = query_ys[:, plot_output_dim_num].unsqueeze(-1) # plot one dimension
query_ys = query_ys.gather(dim=-2, index=indicies)

# Create the figure and axis for the animation
fig, ax = plt.subplots()
true_line, = ax.plot(query_xs.cpu(), query_ys.cpu(), label="True", color="blue")
# Initialize predicted line with the first iterate
y_hat_initial = model.predict(query_xs, subsampled_iterates[0]).cpu()
# plot one dimension
pred_line, = ax.plot(query_xs.cpu(), y_hat_initial, label="Predicted", color="orange")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid()

# Update function using subsampled iterates
def update(frame):
    ax.set_title(f"RLS Iteration {frame * subsample_factor}")
    y_hat = model.predict(query_xs, subsampled_iterates[frame]).cpu()
    # plot one dimension

    pred_line.set_ydata(y_hat)
    return pred_line, ax.title

ani = animation.FuncAnimation(fig, update, frames=len(subsampled_iterates),
                                interval=100, blit=True, repeat_delay=500)

# make the log directory
logdir = "./logs/fixed_polynomial_basis"
import os
if not os.path.exists(logdir):
    os.makedirs(logdir)
# Save the animation
ani.save(f"{logdir}/rls_func_{function_idx}.mp4", writer="ffmpeg")
