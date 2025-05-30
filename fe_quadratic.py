
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.animation as animation

from FunctionEncoder import MultiDimQuadraticDataset, MSECallback, ListCallback, TensorboardCallback, DistanceCallback
from FunctionEncoder import FunctionEncoder
import argparse

from RLS.rls import *

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=3)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--load_path", type=str, default='logs/quadratic_example/least_squares/shared_model/2025-05-30_15-09-26')
# parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
parser.add_argument("--parallel", action="store_true")
args = parser.parse_args()

# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda" if torch.cuda.is_available() else "cpu"
train_method = args.train_method
seed = args.seed
load_path = args.load_path
residuals = args.residuals
if load_path is None:
    logdir = f"logs/quadratic_example/{train_method}/{'shared_model' if not args.parallel else 'parallel_models'}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path
arch = "MLP" if not args.parallel else "ParallelMLP"

# seed torch
torch.manual_seed(seed)

# create a dataset
n_output_dim = 2
if residuals:
    a_range = (0, 3/50) # this makes the true average function non-zero
else:
    a_range = (-3/50, 3/50)
b_range = (-3/50, 3/50)
c_range = (-3/50, 3/50)
input_range = (-10, 10)
dataset = MultiDimQuadraticDataset(a_range=a_range, b_range=b_range, c_range=c_range, input_range=input_range, n_output_dim=n_output_dim)


if load_path is None:
    # create the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            method=train_method,
                            use_residuals_method=residuals).to(device)
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    # create callbacks
    cb1 = TensorboardCallback(logdir) # this one logs training data
    cb2 = DistanceCallback(dataset, tensorboard=cb1.tensorboard) # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    # load the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            method=train_method,
                            use_residuals_method=residuals).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))

# RLS with subsampled iterates for efficiency
    
with torch.no_grad():
    example_xs, example_ys, query_xs, query_ys, sample_info = dataset.sample()
    function_idx = 3  # choose one of the functions
    # print(f'Function {function_idx} A: {sample_info["As"][function_idx]}, B: {sample_info["Bs"][function_idx]}, C: {sample_info["Cs"][function_idx]}')
    n = 100
    example_xs = example_xs[function_idx, :n]
    example_ys = example_ys[function_idx, :n]
    query_xs   = query_xs[function_idx, :n]
    query_ys   = query_ys[function_idx, :n]

    # Load RLS model
    theta0 = torch.zeros((n_basis), device=device)  # initial coefficients
    rls = make_rls(method='standard', n_regressors=n_basis, model=model, theta0=theta0)

    plot_output_dim_num = 1 if n_output_dim > 1 else 0

    parameter_iterates = [rls.theta.clone()]
    for i in range(example_xs.shape[1]):
        x = example_xs[i,:].to(device)
        y = example_ys[i,:].to(device)
        theta, info = rls.update(x, y)
        parameter_iterates.append(theta.clone())

    true_coefficients, G = model.compute_representation(
        # example_xs.unsqueeze(0).to(device), example_ys.unsqueeze(0).to(device)
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
    query_ys = query_ys.squeeze(0)[:, plot_output_dim_num].unsqueeze(-1) # plot one dimension
    query_ys = query_ys.gather(dim=-2, index=indicies)
    
    # Create the figure and axis for the animation
    fig, ax = plt.subplots()
    true_line, = ax.plot(query_xs.cpu(), query_ys.cpu(), label="True", color="blue")
    # Initialize predicted line with the first iterate
    y_hat_initial = model.predict(query_xs.unsqueeze(0), subsampled_iterates[0].unsqueeze(0)).cpu()
    # plot one dimension
    y_hat_initial = y_hat_initial.squeeze(0)[:, plot_output_dim_num].unsqueeze(-1)
    pred_line, = ax.plot(query_xs.cpu(), y_hat_initial, label="Predicted", color="orange")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid()

    # Update function using subsampled iterates
    def update(frame):
        ax.set_title(f"RLS Iteration {frame * subsample_factor}")
        y_hat = model.predict(query_xs.unsqueeze(0), subsampled_iterates[frame].unsqueeze(0)).cpu()
        # plot one dimension
        y_hat = y_hat.squeeze(0)[:, plot_output_dim_num].unsqueeze(-1)
        pred_line.set_ydata(y_hat)
        return pred_line, ax.title

    ani = animation.FuncAnimation(fig, update, frames=len(subsampled_iterates),
                                  interval=100, blit=True, repeat_delay=500)
    ani.save(f"{logdir}/rls_func_{function_idx}.mp4", writer="ffmpeg")
