import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

def lorenz(x, sigmas, rhos, betas):
    x1 = x[..., 0]
    x2 = x[..., 1]
    x3 = x[..., 2]
    dx1 = sigmas * (x2 - x1)
    dx2 = x1 * (rhos - x3) - x2
    dx3 = x1 * x2 - betas * x3
    return torch.stack([dx1, dx2, dx3], dim=-1)

def rk4_difference_only(x0:torch.tensor, dt:torch.tensor,  sigmas:torch.tensor, rhos:torch.tensor, betas:torch.tensor) -> torch.tensor:
    k1 = lorenz(x0, sigmas, rhos, betas)
    k2 = lorenz(x0 + 0.5 * dt * k1, sigmas, rhos, betas)
    k3 = lorenz(x0 + 0.5 * dt * k2, sigmas, rhos, betas)
    k4 = lorenz(x0 + dt * k3, sigmas, rhos, betas)
    return (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Parameters
T = 5_000        # Total steps
dt = 0.01
sigma = 10.0
rho = 28.0
beta = 8/3.0
initial_conditions = torch.tensor([0., 1., 1.05]).unsqueeze(0)
n_frames = 250

# Simulate
states = [initial_conditions]
for _ in range(T):
    current_state = states[-1]
    delta_X = rk4_difference_only(current_state, dt, sigma, rho, beta)
    states.append(current_state + delta_X)
states = torch.cat(states, dim=0)  # Shape (T+1, 2)

ax = plt.figure().add_subplot(projection='3d')

ax.plot(*states.T, lw=0.6)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

# # Downsample
# downsample = T // n_frames
# x_vals = states[::downsample, 0]
# u_vals = states[::downsample, 1]
# t = np.arange(len(x_vals)) * dt * downsample

# # Plot setup
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# line1, = ax1.plot([], [], 'b-', label='Position')
# line2, = ax2.plot([], [], 'r-', label='Velocity')
# ax1.set_xlim(0, t[-1])
# ax1.set_ylim(-1.5, 1.5)
# ax2.set_xlim(0, t[-1])
# ax2.set_ylim(-1.5, 1.5)
# ax1.set_xlabel("Time (s)")
# ax2.set_xlabel("Time (s)")
# ax1.legend()
# ax2.legend()

# # Update function
# def update(frame):
#     line1.set_data(t[:frame+1], x_vals[:frame+1])
#     line2.set_data(t[:frame+1], u_vals[:frame+1])
#     current_time = float(t[frame])  
#     ax1.set_title(f"Position (t={current_time:.2f}s)")
#     ax2.set_title(f"Velocity (t={current_time:.2f}s)")
#     return line1, line2

# # Animate
# ani = animation.FuncAnimation(fig, update, frames=len(t), interval=50, blit=False)
# plt.tight_layout()
# # Save the animation
# # ani.save("mass_spring_damper.gif", writer='imagemagick', fps=30)
# # plt.show()
# ani.save("spring_damper.gif", writer='pillow', fps=20)