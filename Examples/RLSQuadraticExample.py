from datetime import datetime

import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.animation as animation

from FunctionEncoder import QuadraticDataset, RecursiveFunctionEncoder, MSECallback, ListCallback, TensorboardCallback, DistanceCallback

import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--load_path", type=str, default=None)
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
    logdir = f"logs/rls_example/{train_method}/{'shared_model' if not args.parallel else 'parallel_models'}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path
arch = "MLP" if not args.parallel else "ParallelMLP"

# seed torch
torch.manual_seed(seed)

# create a dataset
if residuals:
    a_range = (0, 3/50) # this makes the true average function non-zero
else:
    a_range = (-3/50, 3/50)
b_range = (-3/50, 3/50)
c_range = (-3/50, 3/50)
input_range = (-10, 10)
dataset = QuadraticDataset(a_range=a_range, b_range=b_range, c_range=c_range, input_range=input_range)

# RLS parameters 
init_coefficients = torch.zeros((n_basis, 1), device=device)
forgetting_factor = 1
delta = 1e3

if load_path is None:
    # create the model
    model = RecursiveFunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            method=train_method,
                            use_residuals_method=residuals,
                            forgetting_factor=forgetting_factor,
                            delta=delta,
                            init_coefficients=init_coefficients).to(device)
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
    model = RecursiveFunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            method=train_method,
                            use_residuals_method=residuals,
                            forgetting_factor=forgetting_factor,
                            delta=delta,
                            init_coefficients=init_coefficients).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))

# RLS with subsampled iterates for efficiency
with torch.no_grad():
    example_xs, example_ys, query_xs, query_ys, sample_info = dataset.sample()
    function_idx = 3  # choose one of the functions
    print(f'Function {function_idx} A: {sample_info["As"][function_idx]}, B: {sample_info["Bs"][function_idx]}, C: {sample_info["Cs"][function_idx]}')
    n = 100
    example_xs = example_xs[function_idx, :n]
    example_ys = example_ys[function_idx, :n]
    query_xs   = query_xs[function_idx, :n]
    query_ys   = query_ys[function_idx, :n]

    coefficient_iterates = [init_coefficients.clone()]
    for i in range(example_xs.shape[0]):
        x = example_xs[i].to(device)
        y = example_ys[i].to(device)
        coefficients, info = model.recursive_update(x, y)
        coefficient_iterates.append(coefficients.clone())

    true_coefficients, G = model.compute_representation(
        example_xs.unsqueeze(0).to(device), example_ys.unsqueeze(0).to(device)
    )
    distance = torch.norm(coefficient_iterates[-1] - true_coefficients, p=2)
    print(f"Distance from iterative solution to the true solution: {distance.item()}")

    # Subsample the coefficient iterates to reduce the number of frames
    subsample_factor = 1  # Adjust this factor as needed
    subsampled_iterates = coefficient_iterates[::subsample_factor]
    query_xs, indicies = torch.sort(query_xs, dim=-2)
    query_ys = query_ys.gather(dim=-2, index=indicies)
    
    # Create the figure and axis for the animation
    fig, ax = plt.subplots()
    true_line, = ax.plot(query_xs.cpu(), query_ys.cpu(), label="True", color="blue")
    # Initialize predicted line with the first iterate
    y_hat_initial = model.predict(query_xs.unsqueeze(0), subsampled_iterates[0].T).squeeze(0).cpu()
    pred_line, = ax.plot(query_xs.cpu(), y_hat_initial, label="Predicted", color="orange")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid()

    # Update function using subsampled iterates
    def update(frame):
        ax.set_title(f"RLS Iteration {frame * subsample_factor}")
        y_hat = model.predict(query_xs.unsqueeze(0), subsampled_iterates[frame].T).squeeze(0).cpu()
        pred_line.set_ydata(y_hat)
        return pred_line, ax.title

    ani = animation.FuncAnimation(fig, update, frames=len(subsampled_iterates),
                                  interval=100, blit=True, repeat_delay=500)
    ani.save(f"{logdir}/rls.mp4", writer="ffmpeg")
