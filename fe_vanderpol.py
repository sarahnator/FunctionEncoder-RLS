


from datetime import datetime

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from FunctionEncoder import QuadraticDataset, FunctionEncoder, MSECallback, ListCallback, TensorboardCallback, \
    DistanceCallback

import argparse

from FunctionEncoder.Dataset.VanDerPolDataset import VanDerPolDataset

from RLS.rls import *

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--load_path", type=str, default='logs/vanderpol_example/least_squares/2025-05-26_16-13-20')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
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
    logdir = f"logs/vanderpol_example/{train_method}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path
arch = "NeuralODE"

# seed torch
torch.manual_seed(seed)

# create a dataset
dataset = VanDerPolDataset()

if load_path is None:
    # create the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            method=train_method,
                            use_residuals_method=residuals,
                            model_kwargs={"dt": dataset.dt}
                            ).to(device)
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


# plot
with torch.no_grad():
    # get data
    example_xs, example_ys, query_xs, query_ys, info = dataset.sample()

    # compute coefficients of the first 9 van der pol systems
    batch_size = 9
    coeffs, _ = model.compute_representation(example_xs, example_ys)
    coeffs = coeffs[:batch_size]

    # sample initial states from dataset
    initial_states = torch.rand(batch_size, 2, device=device) * (dataset.input_max - dataset.input_min) + dataset.input_min
    all_states, all_estimated_states = [initial_states], [initial_states]

    # integrate to find outputs of both ground truth and estimate
    for i in trange(1000, desc="Integrating trajectory"):
        # do real first
        current_gt_state = all_states[-1]
        change_in_state = dataset.rk4_difference_only(current_gt_state.unsqueeze(1), dataset.dt, info["mus"][:batch_size]).squeeze(1)
        new_gt_state = current_gt_state + change_in_state
        all_states.append(new_gt_state)

        # do estimate
        current_estimated_state = all_estimated_states[-1]
        change_in_estimated_state = model.predict(current_estimated_state.unsqueeze(1), coeffs).squeeze(1)
        new_estimated_state = current_estimated_state + change_in_estimated_state
        all_estimated_states.append(new_estimated_state)
    all_states = torch.stack(all_states, dim=1)
    all_estimated_states = torch.stack(all_estimated_states, dim=1)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(batch_size):
        ax = axs[i // 3, i % 3]
        ax.plot(all_states[i, :, 0].cpu(), all_states[i, :, 1].cpu(), label="True")
        ax.plot(all_estimated_states[i, :, 0].cpu(), all_estimated_states[i, :, 1].cpu(), label="Estimate")
        ax.set_title(f"mu={info['mus'].flatten()[i].item():0.1f}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(dataset.input_min[0].item(), dataset.input_max[0].item())
        ax.set_ylim(dataset.input_min[1].item(), dataset.input_max[1].item())
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{logdir}/plot.png")
    plt.close()


    # Load RLS model
    n_systems = 1
    for j in range(n_systems):
        theta0 = torch.zeros((n_basis), device=device)  # initial coefficients
        rls = make_rls(method='standard', n_regressors=n_basis, model=model, theta0=theta0)

        true_parameters = coeffs[j].clone().to(device)  # true parameters for the j-th system
        parameter_iterates = [rls.theta.clone()]
        n_iterates = example_xs.shape[1]  # number of iterations to run RLS
        for i in range(n_iterates):
            x = example_xs[j, i, :].to(device)
            y = example_ys[j, i, :].to(device)
            # update RLS
            theta, rls_info = rls.update(x, y)
            parameter_iterates.append(theta.clone())
        parameter_iterates = torch.stack(parameter_iterates, dim=0) # shape (t, n_basis)
        parameter_iterate_norms = torch.norm(parameter_iterates, dim=-1)

        plt.plot(parameter_iterate_norms.cpu())
        plt.yscale("log")
        plt.title("Parameter norms over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Parameter norm")
        plt.show()
        plt.close()
        
        # compute residuals
        residuals = parameter_iterates - true_parameters
        residual_norms = residuals.norm(dim=-1)
        plt.plot(residual_norms.cpu())
        plt.yscale("log")
        plt.title("Parameter Residual norms over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Residual norm")
        plt.show()
        plt.close()

        print(f"Final parameters for system {j}: {parameter_iterates[-1].cpu().numpy()}")
        print(f"True parameters for system {j}: {true_parameters.cpu().numpy()}")
        print(f"Final residuals for system {j}: {residuals[-1].cpu().numpy()}")
        print(f"Final residual norm for system {j}: {residual_norms[-1].cpu().item()}")


        # integrate to find outputs of rls estimate
        rls_state_estimates = [initial_states[j].clone().unsqueeze(0)]  # start with the initial state
        for i in trange(1000, desc="Integrating trajectory"):
            # do the same for the RLS model
            current_rls_state = rls_state_estimates[-1] 
            change_in_rls_state = model.predict(current_rls_state.unsqueeze(1), parameter_iterates[-1].unsqueeze(0)).squeeze(1)
            new_rls_state = current_rls_state + change_in_rls_state
            rls_state_estimates.append(new_rls_state)

        all_rls_states = torch.stack(rls_state_estimates, dim=0)
        # plot the system
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(all_states[j, :, 0].cpu(), all_states[j, :, 1].cpu(), label="True")
        ax.plot(all_estimated_states[j, :, 0].cpu(), all_estimated_states[j, :, 1].cpu(), label="Estimate")
        ax.set_title(f"mu={info['mus'].flatten()[j].item():0.1f}")

        ax.plot(all_rls_states[:,:, 0].cpu(), all_rls_states[:,:, 1].cpu(), label="RLS Estimate")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{logdir}/plot_vdp_{j}.png")


