
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
parser.add_argument("--n_basis", type=int, default=5)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_000)
# parser.add_argument("--load_path", type=str, default='logs/quadratic_example/least_squares/shared_model/2025-05-30_15-09-26')
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
    n = example_xs.shape[1]  # number of examples
    example_xs = example_xs[function_idx, :n]
    example_ys = example_ys[function_idx, :n]
    query_xs   = query_xs[function_idx, :n]
    query_ys   = query_ys[function_idx, :n]

    # Load RLS model
    theta0 = torch.zeros((n_basis), device=device)  # initial coefficients
    rls = make_rls(method='standard', n_regressors=n_basis, model=model, theta0=theta0)

    # plot_output_dim_num = 1 if n_output_dim > 1 else 0

    parameter_iterates = [rls.theta.clone()]
    debug_info = []
    for i in range(example_xs.shape[0]):
        x = example_xs[i,:].to(device)
        y = example_ys[i,:].to(device)
        theta, info = rls.update(x, y)
        parameter_iterates.append(theta.clone())
        debug_info.append(info)
        if torch.norm(info['K']) < 1e-6:
            print(f"Early stopping at iteration {i} due to small Kalman gain.")
            break

    parameter_iterates = torch.stack(parameter_iterates) # shape (n_iterates, n_regressors)
    true_coefficients, G = model.compute_representation(
        # example_xs.unsqueeze(0).to(device), example_ys.unsqueeze(0).to(device)
        example_xs.to(device), example_ys.to(device)
    )

    # compute parameter space error
    true_coefficients = true_coefficients.view(-1)
    rls_coefficients = parameter_iterates[-1].view(-1)
    parameter_distance = torch.norm(rls_coefficients.cpu() - true_coefficients.cpu(), p=2)
    print(f"Distance from iterative solution to the FE solution: {parameter_distance.item()}")
    print(f"FE coefficients: {true_coefficients.cpu().numpy()}")
    print(f"RLS coefficients: {rls_coefficients.cpu().numpy()}")

    # Now evaluate the measurement space error for each output dimension

    query_xs, indicies = torch.sort(query_xs.squeeze(0), dim=-2)
    # expand indices to match the number of output dimensions
    indicies_expanded = indicies.expand(-1, n_output_dim)  # [N_points, 1] -> [N_points, n_output_dim]
    query_ys = query_ys.gather(dim=-2, index=indicies_expanded)

    # compute measurement space error
    y_hat_rls = model.predict(query_xs.unsqueeze(0).to(device), representations=parameter_iterates[-1].unsqueeze(0)).squeeze(0)
    y_hat_fe = model.predict(query_xs.unsqueeze(0).to(device), representations=true_coefficients.unsqueeze(0)).squeeze(0)
    rls_measurement_err = torch.norm((query_ys.cpu() - y_hat_rls.cpu()), dim=0).detach().cpu()
    fe_measurement_err = torch.norm((query_ys.cpu() - y_hat_fe.cpu()), dim=0).detach().cpu()
    print(f"RLS measurement error: {rls_measurement_err}")
    print(f"FE measurement error: {fe_measurement_err}")

    # Subsample the coefficient iterates to reduce the number of frames if needed
    subsample_factor = 1  # No subsampling if set to 1
    subsampled_iterates = parameter_iterates[::subsample_factor]

    # plot the results for each output dimension
    fig, axs = plt.subplots(n_output_dim, 1, sharex=True, figsize=(6, 2*n_output_dim))

    # For each dimension d, plot the true curve and then initialize a predicted line:
    true_lines = []
    pred_lines = []
    # Compute an initial prediction using the very first iterate:
    y_hat_init = model.predict(query_xs.unsqueeze(0),
                                   subsampled_iterates[0].unsqueeze(0)
                                  ).squeeze(0).detach()  # [N_points, n_output_dim]
    for d, ax in enumerate(axs):
        # Plot the true data in blue
        true_line, = ax.plot(query_xs.cpu().numpy(),
                             query_ys[:, d].cpu().numpy(),
                            #  label=f"True (dim {d})",
                             label=f"True",
                             color="C0")
        true_lines.append(true_line)

        # Compute an initial prediction using the very first iterate:
        y_hat_init_d = y_hat_init[:, d].cpu().numpy()  # take the d-th column

        # Plot the predicted curve in orange (for RLS iterates)
        pred_line, = ax.plot(query_xs.cpu().numpy(),
                             y_hat_init_d,
                            #  label=f"RLS pred (dim {d})",
                             label=f"RLS pred",
                             color="C1")
        pred_lines.append(pred_line)

        ax.set_ylabel(f"y[{d}]")
        ax.grid(True)

    axs[-1].set_xlabel("x")
    handles, labels = axs[0].get_legend_handles_labels()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.subplots_adjust(top=0.9)  # Increase the top margin (try values between 0.9 and 1)
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    
    def update(frame):
        the_iter = subsampled_iterates[frame].unsqueeze(0)
        y_hat_all = model.predict(query_xs.unsqueeze(0), the_iter).squeeze(0).detach()  # [N_points, n_output_dim]
        for d, pred_line in enumerate(pred_lines):
            new_y = y_hat_all[:, d].cpu().numpy()
            pred_line.set_ydata(new_y)
        for ax in axs:
            if ax == axs[0]: # only set title for the first axis
                ax.set_title(f"RLS iteration {frame * subsample_factor}")
        return (*pred_lines, *axs)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(subsampled_iterates),
        interval=100,
        blit=False,
        repeat_delay=500
    )

    ani.save(f"{logdir}/rls_func.mp4", writer="ffmpeg")

    # other plots

    # plot the kalman gain over iterations
    rls.plot_kalman_gain(
        debug_info=debug_info,
        logdir=logdir
    )
    # kalman_gains = torch.stack([info['K'] for info in debug_info], dim=0)  # shape (n_iterates, n_regressors, n_outputs)
    # normed_kalman_gains = torch.norm(kalman_gains, dim=-2)  # shape (n_iterates, n_regressors)
    # # plot the norm of the kalman gain over iterations for each output dimension
    # fig, axs = plt.subplots(n_output_dim, 1, sharex=True, figsize=(6, 2*n_output_dim))
    # for d in range(n_output_dim):
    #     axs[d].plot(normed_kalman_gains[:, d].cpu().numpy())
    #     axs[d].set_ylabel(f"||K||, dim {d}")
    #     axs[d].grid(True)
    # axs[-1].set_xlabel("RLS iteration")
    # # set log scale for y-axis
    # for ax in axs:
    #     ax.set_yscale('log')
    # plt.tight_layout()
    # plt.savefig(f"{logdir}/kalman_gain.png")
    # plt.clf()

    # plot the condition number of the covariance matrix over iterations
    rls.plot_condition_number(
        debug_info=debug_info,
        logdir=logdir
    )
    # condition_numbers = torch.stack([torch.linalg.cond(info['P_new']) for info in debug_info], dim=0)  # shape (n_iterates,)
    # plt.plot(condition_numbers.cpu().numpy())
    # plt.xlabel("RLS iteration")
    # plt.ylabel("Condition number of covariance matrix")
    # plt.title("Condition number of covariance matrix over iterations")
    # plt.grid(True)
    # # set log scale for y-axis
    # plt.yscale('log')
    # plt.tight_layout()
    # plt.savefig(f"{logdir}/condition_number.png")
    # plt.clf()

    # plot the measurement space error over iterations
    rls.plot_measurement_space_error(
        query_xs=query_xs,
        query_ys=query_ys,
        parameter_iterates=parameter_iterates,
        logdir=logdir
    )
    # all_y_hat_rls = [model.predict(query_xs.unsqueeze(0).to(device), representations=theta.unsqueeze(0)).squeeze(0).detach() for theta in parameter_iterates]
    # all_y_hat_rls = torch.stack(all_y_hat_rls, dim=0)  # shape (n_iterates, N_points, n_output_dim)
    # measurement_errors = all_y_hat_rls - query_ys.unsqueeze(0)  # shape (n_iterates, N_points, n_output_dim)
    # normed_measurement_errors = torch.norm(measurement_errors, dim=-2)  # shape (n_iterates, n_output_dim)
    # # plot the norm of the measurement space error over iterations for each output dimension
    # fig, axs = plt.subplots(n_output_dim, 1, sharex=True, figsize=(6, 2*n_output_dim))
    # for d in range(n_output_dim):
    #     axs[d].plot(normed_measurement_errors[:, d].cpu().numpy())
    #     axs[d].set_ylabel(f"||e||, dim {d}")
    #     axs[d].grid(True)
    # axs[-1].set_xlabel("RLS iteration")
    # # set log scale for y-axis
    # for ax in axs:
    #     ax.set_yscale('log')
    # plt.title("Measurement space error over iterations")
    # plt.tight_layout()
    # plt.savefig(f"{logdir}/measurement_space_errors.png")
    # plt.clf()

    # plot the parameter space error over iterations
    rls.plot_parameter_space_error(
        parameter_iterates, 
        true_coefficients,
        logdir=logdir
    )
    # parameter_space_errors = torch.norm(parameter_iterates - true_coefficients.unsqueeze(0), dim=-1)  # shape (n_iterates,)
    # plt.plot(parameter_space_errors.cpu().numpy())
    # plt.xlabel("RLS iteration")
    # plt.ylabel("Parameter space error (L2 norm)")
    # plt.title("Parameter space error over iterations")
    # plt.grid(True)
    # plt.yscale('log')  # log scale for better visibility
    # plt.tight_layout()
    # plt.savefig(f"{logdir}/parameter_space_error.png")
    # plt.clf()