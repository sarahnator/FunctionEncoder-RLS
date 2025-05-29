



from datetime import datetime

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from FunctionEncoder import QuadraticDataset, FunctionEncoder, MSECallback, ListCallback, TensorboardCallback, \
    DistanceCallback
from FunctionEncoder import GeneralizedWRRLSFunctionEncoder as WRRLSFunctionEncoder

import argparse

from FunctionEncoder.Dataset.VanDerPolDataset import VanDerPolDataset

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
    # RLS parameters
    init_coefficients = torch.zeros((n_basis, 1), device=device)
    forgetting_factor = 1
    alpha = 1e-3
    model = WRRLSFunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            method=train_method,
                            use_residuals_method=residuals,
                            forgetting_factor=forgetting_factor,
                            alpha=alpha,
                            init_coefficients=init_coefficients).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))

# plot
with torch.no_grad():
    # get data
    example_xs, example_ys, query_xs, query_ys, info = dataset.sample()

    # compute coefficients of the first 9 van der pol systems
    batch_size = 1
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

    # recursive Least Squares (RLS) to estimate the coefficients
    batch_init_coefficients = init_coefficients.reshape(1, -1).clone()
    # batch_init_coefficients = init_coefficients.clone()
    # .reshape(1, -1)
    # copy for all n_functions
    batch_init_coefficients = batch_init_coefficients.expand(batch_size, -1, -1)
    model.coefficients = batch_init_coefficients.clone().to(device)
    model.Q_tilde = model.Q_tilde.unsqueeze(0).expand(batch_size, n_basis, -1).to(device)
    coefficient_iterates = [batch_init_coefficients]
    for i in range(example_xs.shape[1]):
        # x = example_xs[0, i, :].to(device)
        # y = example_ys[0, i, :].to(device)
        x = example_xs[:batch_size, i, :].to(device)
        y = example_ys[:batch_size, i, :].to(device)
        coefficients, info = model.recursive_update(x, y)
        coefficient_iterates.append(coefficients.clone())
    coefficient_iterates = torch.stack(coefficient_iterates, dim=0).squeeze(-1)
    coef_iterate_norms = torch.norm(coefficient_iterates, dim=-1)
    residuals = coefficient_iterates - coeffs
    residual_norms = residuals.norm(dim=1)
    plt.plot(coef_iterate_norms.cpu())
    plt.yscale("log")
    plt.title("Coefficient norms over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Coefficient norm")
    plt.show()

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(9):
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

