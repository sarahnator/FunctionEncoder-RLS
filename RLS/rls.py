import torch
import matplotlib.pyplot as plt

"""
rls.py

Several variants of Recursive Least Squares:
 - Plain RLS
 - Forgetting‐factor RLS
 - Weighted RLS
 - LMS-style RLS
#  - Regularized RLS
#  - Kalman‐filter extension
"""

import torch
from abc import ABC, abstractmethod
from FunctionEncoder.Model.FunctionEncoder import FunctionEncoder
import torch

class PolynomialModel():
    """
    Polynomial regression model for univariate inputs. This is to test the RLS algorithms.
    """
    def __init__(self, input_size: int, output_size: int, degree: int = 2):
        """
        degree: highest power of x to include (so number of regressors = degree+1)
        """
        self.degree = degree
        self.input_size = input_size
        self.output_size = output_size
        # Ensure input_size is 1 for polynomial regression
        if input_size[0] != 1:
            raise ValueError("PolynomialModel only supports input_size=1 for univariate polynomial regression.")
        # Ensure output_size is 1 for univariate polynomial regression
        if output_size[0] != 1:
            raise ValueError("PolynomialModel only supports output_size=1 for univariate polynomial regression.")

    def forward_basis_functions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build the design matrix Phi for unbatched inputs.
        
        Args:
            x: Tensor of shape (n_points,) or (n_points,1)
        
        Returns:
            Phi: Tensor of shape (n_points, degree+1),
                 where column j is x**j.
        """
        # Flatten to (n_points,)
        x = x.view(-1)
        # Stack [1, x, x^2, ..., x^degree]
        powers = [x**d for d in range(self.degree+1)]
        Phi = torch.stack(powers, dim=-1)  # → (n_points, degree+1)
        return Phi.cuda() if x.is_cuda else Phi  # Move to GPU if needed

    def predict(self, x: torch.Tensor, representations: torch.Tensor) -> torch.Tensor:
        """
        Compute y_hat = Phi(x) @ theta.
        
        Args:
            x:           Tensor of shape (n_points,) or (n_points,1)
            representations: Tensor of shape (degree+1)

        Returns:
            y_hat: Tensor of shape (output_dim,)
        """
        Phi = self.forward_basis_functions(x)       # (n_points, r)
        y_hat = Phi @ representations                  # (n_points,)
        return y_hat

    def compute_representation(self, x: torch.Tensor, y: torch.Tensor):
        """
        Solve the ordinary least-squares coefficients.
        
        Args:
            x: Tensor of shape (n_points,) or (n_points,1)
            y: Tensor of shape (n_points,)

        Returns:
            theta:   Tensor of shape (degree+1,)
            sol:     the full LSTSQ result (with .solution, .residuals, etc.)
        """
        Phi = self.forward_basis_functions(x)       # (n_points, r)
        sol = torch.linalg.lstsq(Phi, y.view(-1))   # solves Phi @ theta = y
        return sol.solution, sol
    
class BaseRLS(ABC):
    """
    Abstract base class. Holds common state: theta (coefficients), P (covariance).
    """
    def __init__(self, n_regressors, model, P0=None, theta0=None):
        self.n = n_regressors
        self.model = model
        self.theta = theta0 if theta0 is not None else torch.zeros((n_regressors))

        self.P = P0 if P0 is not None else torch.eye(n_regressors) * 1e0
        # move to device if model is on GPU
        # if next(self.model.parameters()).is_cuda:
        self.theta = self.theta.cuda()
        self.P = self.P.cuda()

    def update_matrix_factorization(self, P_prev, Phi):
        """
        Square‐root RLS update (Morf & Kailath, Ljung §11.7):
        Given the previous covariance P_prev (d×d) and the new regressor Phi (m×d),
        produce the updated P_new in an O((p+d)^3) QR step without ever inverting P.
        """
        # number of outputs (p) and number of regressors (d)
        p = self.model.output_size[0]
        d = self.n
        device = P_prev.device
        dtype  = P_prev.dtype

        # 1. Cholesky on P_prev to get lower‐triangular Q_prev  (d×d)
        Q_prev = torch.linalg.cholesky(P_prev)   # P_prev = Q_prev @ Q_prev.T

        # 2. Build p(t) = Cholesky of [k·R] or of R if no forgetting factor
        if hasattr(self, 'R'):
            S    = self.R.clone().to(device=device, dtype=dtype)
            lamb = 1.0
        else:
            k    = getattr(self, 'forgeting_factor', 1.0)
            S    = k * torch.eye(p, device=device, dtype=dtype)
            lamb = k
        mu_t = torch.linalg.cholesky(S)  # lower‐triangular square‐root of k·R (or R itself)

        # 3. Form the (p+d)×(p+d) block matrix:
        #      L = [   mu_t       |   0_p×d ;
        #            Q_prev.T @ Phi.T  |  Q_prev.T ]
        zero_top = torch.zeros((p, d), device=device, dtype=dtype)
        top      = torch.cat((mu_t, zero_top), dim=1)          # shape (p×(p+d))

        bottom   = torch.cat((Q_prev.T @ Phi.T, Q_prev.T), dim=1)  # shape (d×(p+d))
        L_full   = torch.cat((top, bottom), dim=0)                  # shape ((p+d)×(p+d))

        # 4. Single QR → L_full = Q_full @ R_full, R_full upper triangular
        #    We only need R_full, so do a “reduced” QR:
        # From pyTorch docs: The reduced QR decomposition agrees with the full QR decomposition when n >= m (wide matrix).
        Q_full, R_full = torch.linalg.qr(L_full, mode='reduced')
        # Now R_full is (p+d)×(p+d) upper triangular.
        # R_full is the block matrix:
        #      R_full = [ Pi.T | L_tilde.T ;
        #                 0_d×p | Q_bar.T ]

        # 5. The new Cholesky factor Q_new is the lower‐right d×d block of R_full
        R_block = R_full[p:, p:]  # pick out the bottom‐right d×d submatrix

        # 6. Reconstruct P_new = (R_block)^T @ (R_block), then divide by lamb if needed
        P_new = R_block.T @ R_block
        if (lamb != 0.0) and (lamb != 1.0):
            P_new = P_new / lamb

        # We can compute the gain K and the innovation covariance S if needed
        Pi_T = R_full[:p, :p]  # upper-left p×p block
        L_tilde_T = R_full[:p, p:] # upper-right p×d block 
        # K = (L_tilde_T.mT @ torch.linalg.inv(Pi))       # least‐reliable (forms an explicit inverse)
        # or equivalently
        # K = torch.linalg.solve(Pi, L_tilde.T).T     # stable: solves π X = L̃^T, then transpose
        # Solve π @ X = L_tilde_T without forming π^{-1}, then transpose to get K = L̃ @ π^{-1}, which has shape (d × p)
        X = torch.linalg.solve_triangular(Pi_T, L_tilde_T, upper=True, left=True)
        K = X.T  # shape (p, d)

        S = Pi_T.T @ Pi_T

        # check matrix shapes
        if P_new.dim() != 2 or P_new.shape[0] != self.n or P_new.shape[1] != self.n:
            raise ValueError(f"Invalid covariance matrix shape: {P_new.shape}")
        if K.dim() != 2 or K.shape[0] != self.n or K.shape[1] != self.model.output_size[0]:
            raise ValueError(f"Invalid Kalman gain shape: {K.shape}")
        if S.dim() != 2 or S.shape[0] != self.model.output_size[0] or S.shape[1] != self.model.output_size[0]:
            raise ValueError(f"Invalid innovation covariance shape: {S.shape}")

        return P_new, K, S

            
    def debug_info(self):
        """
        Minimal sanity checks for P in RLS:

        1) P should be symmetric.
        2) P should be positive-definite (Cholesky succeeds).
        3) Cond(P) should not be astronomically large.
        4) Diagonal entries should not collapse to near zero.
        """
        # print(self.P)

        # (1) Symmetry check:
        if not torch.allclose(self.P, self.P.T, atol=1e-6):
            print("P is not symmetric.")

        # (2) Positive-definiteness check via Cholesky:
        try:
            _ = torch.linalg.cholesky(self.P)
        except RuntimeError:
            print("P is not positive-definite (Cholesky failed).")
            # Early-exit if you want to skip the remaining checks
            return

        # (3) Condition number check:
        cond_P = torch.linalg.cond(self.P)
        if cond_P > 1e8:
            print(f"P is poorly conditioned (cond ≈ {cond_P:.2e}).")

        # (4) Diagonal‐scale check:
        diag_vals = torch.diagonal(self.P)
        max_diag = torch.max(torch.abs(diag_vals))
        tol     = 1e-8 * max_diag    # “tiny” relative to largest variance
        if torch.any(diag_vals < tol):
            idxs = torch.where(diag_vals < tol)[0].tolist()
            print(f"Diagonal entries very small at indices {idxs} (may lose definiteness).")

    def plot_kalman_gain(self, debug_info, logdir='.'):
        """
        Plot the Kalman gain K from the debug_info dictionary.
        """
        kalman_gains = torch.stack([info['K'] for info in debug_info], dim=0)  # shape (n_iterates, n_regressors, n_outputs)
        normed_kalman_gains = torch.norm(kalman_gains, dim=-2)  # shape (n_iterates, n_regressors)
        # plot the norm of the kalman gain over iterations for each output dimension
        n_output_dim = kalman_gains.shape[-1]  # number of output dimensions
        fig, axs = plt.subplots(n_output_dim, 1, sharex=True, figsize=(6, 2*n_output_dim))
        for d in range(n_output_dim):
            axs[d].plot(normed_kalman_gains[:, d].cpu().numpy())
            axs[d].set_ylabel(f"||K||, dim {d}")
            axs[d].grid(True)
        axs[-1].set_xlabel("RLS iteration")
        # set log scale for y-axis
        for ax in axs:
            ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(f"{logdir}/kalman_gain.png")
        plt.clf()

    def plot_condition_number(self, debug_info, logdir='.'):
        """
        Plot the condition number of the covariance matrix P from the debug_info dictionary.
        """
        condition_numbers = torch.stack([torch.linalg.cond(info['P_new']) for info in debug_info], dim=0)  # shape (n_iterates,)
        plt.plot(condition_numbers.cpu().numpy())
        plt.xlabel("RLS iteration")
        plt.ylabel("Condition number of covariance matrix")
        plt.title("Condition number of covariance matrix over iterations")
        plt.grid(True)
        # set log scale for y-axis
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f"{logdir}/condition_number.png")
        plt.clf()

    def plot_measurement_space_error(self, query_xs, parameter_iterates, query_ys, logdir='.'):
        """
        Plot the norm of the measurement space error over iterations.
        This is the error between the predicted and actual measurements.
        
        Args:
            query_xs: Tensor of shape (n_iterates, n_datapoints, input_dim)
            parameter_iterates: Tensor of shape (n_iterates, n_regressors)
            query_ys: Tensor of shape (n_iterates, n_datapoints, output_dim)
            logdir: Directory to save the plot
        """
        all_y_hat_rls = [self.model.predict(query_xs.unsqueeze(0).to('cuda'), representations=theta.unsqueeze(0)).squeeze(0).detach() for theta in parameter_iterates]
        all_y_hat_rls = torch.stack(all_y_hat_rls, dim=0)  # shape (n_iterates, N_points, n_output_dim)
        measurement_errors = all_y_hat_rls - query_ys.unsqueeze(0)  # shape (n_iterates, N_points, n_output_dim)
        normed_measurement_errors = torch.norm(measurement_errors, dim=-2)  # shape (n_iterates, n_output_dim)
        # plot the norm of the measurement space error over iterations for each output dimension
        n_output_dim = normed_measurement_errors.shape[-1]
        fig, axs = plt.subplots(n_output_dim, 1, sharex=True, figsize=(6, 2*n_output_dim))
        for d in range(n_output_dim):
            axs[d].plot(normed_measurement_errors[:, d].cpu().numpy())
            axs[d].set_ylabel(f"||e||, dim {d}")
            axs[d].grid(True)
        axs[-1].set_xlabel("RLS iteration")
        # set log scale for y-axis
        for ax in axs:
            ax.set_yscale('log')
        plt.title("Measurement space error over iterations")
        plt.tight_layout()
        plt.savefig(f"{logdir}/measurement_space_errors.png")
        plt.clf()

    def plot_parameter_space_error(self, parameter_iterates, true_parameters, logdir='.'):
        """
        Plot the norm of the parameter space error over iterations.
        This is the error between the current and previous coefficients.
        
        Args:
            parameter_iterates: Tensor of shape (n_iterates, n_regressors)
            true_parameters: Tensor of shape (n_regressors,) -- the true function encoder coefficients
            logdir: Directory to save the plot
        """
        parameter_space_errors = torch.norm(parameter_iterates - true_parameters.unsqueeze(0), dim=-1)  # shape (n_iterates,)
        plt.plot(parameter_space_errors.cpu().numpy())
        plt.xlabel("RLS iteration")
        plt.ylabel("Parameter space error (L2 norm)")
        plt.title("Parameter space error over iterations")
        plt.grid(True)
        plt.yscale('log')  # log scale for better visibility
        plt.tight_layout()
        plt.savefig(f"{logdir}/parameter_space_error.png")
        plt.clf()

    @abstractmethod
    def update(self, x: torch.tensor, y: torch.tensor, debug_mode=True):
        """
        x: input vector (input_dim,) 
        y: vector measurement (output_dim,) 
        debug_mode: if True, prints debug information
        Returns:
        theta: updated coefficients (n_regressors,)
        info: dictionary with debug information
        """
        pass
    @abstractmethod
    def square_root_update(self,  x: torch.tensor, y: torch.tensor, debug_mode=True):
        """
        Square‐root RLS update (Morf & Kailath, Ljung §11.7):
        Given the previous covariance P_prev (d×d) and the new regressor Phi (m×d),
        produce the updated P_new in an O((p+d)^3) QR step without ever inverting P.
        """
        pass

class StandardRLS(BaseRLS):
    """Plain RLS with optional regularization via initial P."""

    def update(self, x, y, debug_mode=True):
        assert len(x.shape) in [1], f"Expected x to have shape (d,), got {x.shape}"
        assert len(y.shape) in [1], f"Expected y to have shape (m,), got {y.shape}"

        theta_prev = self.theta.clone()  # Store previous coefficients
        P_prev = self.P.clone() 

        # 1. Compute the regressor matrix (basis function representation of the input)
        Phi = self.model.forward_basis_functions(x) # shape (n_outputs, n_regressors)
        if Phi.dim() != 2 or Phi.shape[0] != self.model.output_size[0] or Phi.shape[1] != self.n:
            raise ValueError(f"Invalid regressor shape: {Phi.shape}")

        # 2. Compute the prediction y_hat = Phi @ theta_prev with shape (n_outputs,)
        query_xs = x.unsqueeze(0) if x.dim() == 1 else x.unsqueeze(1) # Ensure query_xs is (n_datapoints, input_dim)
        representations = theta_prev  # shape (n_regressors,)
        if type(self.model) == FunctionEncoder:  
            # FunctionEncoder expects query_xs with shape (batch_size, n_datapoints, input_dim)
            # and representations with shape (batch_size, n_regressors)
            query_xs = query_xs.unsqueeze(0)  # Add batch dimension
            representations = representations.unsqueeze(0)
            y_hat = self.model.predict(query_xs, representations=representations) # returns shape (batch_size, n_datapoints, output_dim)
            y_hat = y_hat.squeeze(0).squeeze(0)  # Remove batch and n_datapoints dimensions
        else:
            # For PolynomialModel, we can directly use the predict method
            y_hat = self.model.predict(query_xs, representations)
        #  Ensure y_hat is (output_dim,)
        if y_hat.dim() != 1 or y_hat.shape[0] != self.model.output_size[0]:
            raise ValueError(f"Invalid prediction shape: {y_hat.shape}")

        # 3. Compute the prediction error (innovation)
        e = y - y_hat # shape (output_dim,)
        if e.shape[-1] != self.model.output_size[0]:
            raise ValueError(f"Invalid prediction error shape: {e.shape}")

        # 4. Compute the Kalman gain K
        # First, Compute the innovation covariance: shape (output_dim, output_dim)
        # S = Phi @ self.P @ Phi.T + torch.eye(self.model.output_size) * 1e-6  # small regularization to avoid singularity
        S = Phi @ self.P @ Phi.T + torch.eye(self.model.output_size[0]).cuda()
        if S.dim() != 2 or S.shape[0] != self.model.output_size[0] or S.shape[1] != self.model.output_size[0]:
            raise ValueError(f"Invalid innovation covariance shape: {S.shape}")    
        # try:
        #     S = torch.linalg.inv(S)  # Invert S to compute Kalman gain
        # except RuntimeError:
        #     raise ValueError("Innovation covariance S is not invertible.")
        # K = self.P @ Phi.T @ S  # expect shape (n_regressors, output_dim)
        K = torch.linalg.solve(S, Phi @ self.P.T).T  # expect shape (output_dim, n_regressors)
        if K.dim() != 2 or K.shape[0] != self.n or K.shape[1] != self.model.output_size[0]:
            raise ValueError(f"Invalid Kalman gain shape: {K.shape}")

        # 4. Update the parameter estimate (coefficients)
        self.theta = self.theta + K @ e # expect shape (n_regressors,)
        if self.theta.dim() != 1 or self.theta.shape[0] != self.n:
            raise ValueError(f"Invalid coefficients shape: {self.theta.shape}")

        # 5. Update the covariance matrix
        self.P = self.P - K @ Phi @ self.P # expect shape (n_regressors, n_regressors)
        if self.P.dim() != 2 or self.P.shape[0] != self.n or self.P.shape[1] != self.n:
            raise ValueError(f"Invalid covariance matrix shape: {self.P.shape}")

        # # Force symmetry and positive definiteness
        # self.P = (self.P + self.P.T) / 2  # Ensure symmetry
        
        # other_P, other_K, other_S = self.update_matrix_factorization(P_prev, Phi)

        if debug_mode:
            self.debug_info()

        info = {
            'theta_prev': theta_prev,
            'P_prev': P_prev,
            'Phi': Phi,
            'y_hat': y_hat,
            'e': e,
            'K': K,
            'P_new': self.P,
            'S': S
        }

        return self.theta, info
    
    def square_root_update(self, x, y, debug_mode=True):
        """
        Square‐root RLS update (Morf & Kailath, Ljung §11.7):
        Given the previous covariance P_prev (d×d) and the new regressor Phi (m×d),
        produce the updated P_new in an O((p+d)^3) QR step without ever inverting P.
        """
        assert len(x.shape) in [1], f"Expected x to have shape (d,), got {x.shape}"
        assert len(y.shape) in [1], f"Expected y to have shape (m,), got {y.shape}"

        theta_prev = self.theta.clone()
        P_prev = self.P.clone()

        # 1. Compute the regressor matrix (basis function representation of the input)
        Phi = self.model.forward_basis_functions(x)  # shape (n_outputs, n_regressors)
        if Phi.dim() != 2 or Phi.shape[0] != self.model.output_size[0] or Phi.shape[1] != self.n:
            raise ValueError(f"Invalid regressor shape: {Phi.shape}")   
        
        # 2. Compute the prediction y_hat = Phi @ theta_prev with shape (n_outputs,)
        query_xs = x.unsqueeze(0) if x.dim() == 1 else x.unsqueeze(1)
        representations = theta_prev  # shape (n_regressors,)
        if type(self.model) == FunctionEncoder:
            # FunctionEncoder expects query_xs with shape (batch_size, n_datapoints, input_dim)
            # and representations with shape (batch_size, n_regressors)
            query_xs = query_xs.unsqueeze(0)
            representations = representations.unsqueeze(0)
            y_hat = self.model.predict(query_xs, representations=representations)  # returns shape (batch_size, n_datapoints, output_dim)
            y_hat = y_hat.squeeze(0).squeeze(0)  # Remove batch and n_datapoints dimensions
        else:
            # For PolynomialModel, we can directly use the predict method
            y_hat = self.model.predict(query_xs, representations)
        # Ensure y_hat is (output_dim,)
        if y_hat.dim() != 1 or y_hat.shape[0] != self.model.output_size[0]:
            raise ValueError(f"Invalid prediction shape: {y_hat.shape}")
        # 3. Compute the prediction error (innovation)
        e = y - y_hat  # shape (output_dim,)
        if e.shape[-1] != self.model.output_size[0]:
            raise ValueError(f"Invalid prediction error shape: {e.shape}")
        
        # use matrix factorization to update P, compute K and S
        P_new, K, S = self.update_matrix_factorization(P_prev, Phi)

        # 4. Update the parameter estimate (coefficients)
        self.theta = self.theta + K @ e  # expect shape (n_regressors,)
        if self.theta.dim() != 1 or self.theta.shape[0] != self.n:
            raise ValueError(f"Invalid coefficients shape: {self.theta.shape}") 
        # Note: P_new is already updated in the update_matrix_factorization method
        self.P = P_new
        if self.P.dim() != 2 or self.P.shape[0] != self.n or self.P.shape[1] != self.n:
            raise ValueError(f"Invalid covariance matrix shape: {self.P.shape}")  

        if debug_mode:
            self.debug_info()

        info = {
            'theta_prev': theta_prev,
            'P_prev': P_prev,
            'Phi': Phi,
            'y_hat': y_hat,
            'e': e,
            'K': K,
            'P_new': self.P,
            'S': S
        }

        return self.theta, info


class ForgettingFactorRLS(BaseRLS):
    """
    RLS with a forgetting factor to decay past observations.
    This is useful for non-stationary environments where older data should have less influence.
    The forgetting factor should be in the range [0.95, 1.0]. 
        - Set to 1.0 for standard RLS to estimate time-invariant (constant) parameters.
        - For time-varying parameters, set to a value less than 1.0. Common values are in the range [0.95, 0.988] (see: https://www.mathworks.com/help/ident/ug/algorithms-for-online-estimation.html).
    This is the multi-variate version of the forgetting factor algroithm here: https://www.mathworks.com/help/ident/ug/forgetting-factor-algorithm.html
    """
    def __init__(self, n_regressors, forgetting_factor=0.998, **kwargs):
        super().__init__(n_regressors, **kwargs)
        self.forgetting_factor = forgetting_factor
        assert 0.95 <= forgetting_factor <= 1.0, "Forgetting factor should be in [0.95, 1.0] range."

    def update(self, x, y, debug_mode=True):
        assert len(x.shape) in [1], f"Expected x to have shape (d,), got {x.shape}"
        assert len(y.shape) in [1], f"Expected y to have shape (m,), got {y.shape}"

        theta_prev = self.theta.clone()  # Store previous coefficients
        P_prev = self.P.clone() 

        # 1. Compute the regressor matrix (basis function representation of the input)
        Phi = self.model.forward_basis_functions(x) # shape (n_outputs, n_regressors)
        if Phi.dim() != 2 or Phi.shape[0] != self.model.output_size[0] or Phi.shape[1] != self.n:
            raise ValueError(f"Invalid regressor shape: {Phi.shape}")

        # 2. Compute the prediction y_hat = Phi @ theta_prev with shape (n_outputs,)
        query_xs = x.unsqueeze(0) if x.dim() == 1 else x.unsqueeze(1) # Ensure query_xs is (n_datapoints, input_dim)
        representations = theta_prev  # shape (n_regressors,)
        if type(self.model) == FunctionEncoder:  
            # FunctionEncoder expects query_xs with shape (batch_size, n_datapoints, input_dim)
            # and representations with shape (batch_size, n_regressors)
            query_xs = query_xs.unsqueeze(0)  # Add batch dimension
            representations = representations.unsqueeze(0)
            y_hat = self.model.predict(query_xs, representations=representations) # returns shape (batch_size, n_datapoints, output_dim)
            y_hat = y_hat.squeeze(0).squeeze(0)  # Remove batch and n_datapoints dimensions
        else:
            # For PolynomialModel, we can directly use the predict method
            y_hat = self.model.predict(query_xs, representations)
        #  Ensure y_hat is (output_dim,)
        if y_hat.dim() != 1 or y_hat.shape[0] != self.model.output_size[0]:
            raise ValueError(f"Invalid prediction shape: {y_hat.shape}")

        # 3. Compute the prediction error (innovation)
        e = y - y_hat # shape (output_dim,)
        if e.shape[-1] != self.model.output_size[0]:
            raise ValueError(f"Invalid prediction error shape: {e.shape}")

        # 4. Compute the Kalman gain K
        # First, Compute the innovation covariance: shape (output_dim, output_dim)
        S = Phi @ self.P @ Phi.T + self.forgetting_factor * torch.eye(self.model.output_size[0]).cuda()
        if S.dim() != 2 or S.shape[0] != self.model.output_size[0] or S.shape[1] != self.model.output_size[0]:
            raise ValueError(f"Invalid innovation covariance shape: {S.shape}")    
        K = torch.linalg.solve(S, Phi @ self.P.T).T  # expect shape (output_dim, n_regressors)
        if K.dim() != 2 or K.shape[0] != self.n or K.shape[1] != self.model.output_size[0]:
            raise ValueError(f"Invalid Kalman gain shape: {K.shape}")

        # 4. Update the parameter estimate (coefficients)
        self.theta = self.theta + K @ e # expect shape (n_regressors,)
        if self.theta.dim() != 1 or self.theta.shape[0] != self.n:
            raise ValueError(f"Invalid coefficients shape: {self.theta.shape}")

        # 5. Update the covariance matrix
        self.P = (1 / self.forgetting_factor) * (self.P - K @ Phi @ self.P) # expect shape (n_regressors, n_regressors)
        if self.P.dim() != 2 or self.P.shape[0] != self.n or self.P.shape[1] != self.n:
            raise ValueError(f"Invalid covariance matrix shape: {self.P.shape}")
        
        if debug_mode:
            self.debug_info()

        info = {
            'theta_prev': theta_prev,
            'P_prev': P_prev,
            'Phi': Phi,
            'y_hat': y_hat,
            'e': e,
            'K': K,
            'P_new': self.P
        }

        return self.theta, info

class WeightedRLS(BaseRLS):
    """RLS with a weight matrix R (covariance) to account for measurement noise or importance of outputs.
    This is useful when different outputs have different levels of noise or importance.
    The weight matrix R should be a diagonal matrix with positive entries.
    If R is None, it defaults to the identity matrix, which is equivalent to standard RLS.
    """
    def __init__(self, n_regressors, R=None, **kwargs):
        super().__init__(n_regressors, **kwargs)
        # R is a diagonal weight matrix (n_outputs, n_outputs) or None for identity
        self.R = R if R is not None else torch.eye(self.model.output_size[0]).cuda()

    def update(self, x, y, debug_mode=True):
        assert len(x.shape) in [1], f"Expected x to have shape (d,), got {x.shape}"
        assert len(y.shape) in [1], f"Expected y to have shape (m,), got {y.shape}"

        theta_prev = self.theta.clone()  # Store previous coefficients
        P_prev = self.P.clone() 

        # 1. Compute the regressor matrix (basis function representation of the input)
        Phi = self.model.forward_basis_functions(x) # shape (n_outputs, n_regressors)
        if Phi.dim() != 2 or Phi.shape[0] != self.model.output_size[0] or Phi.shape[1] != self.n:
            raise ValueError(f"Invalid regressor shape: {Phi.shape}")

        # 2. Compute the prediction y_hat = Phi @ theta_prev with shape (n_outputs,)
        query_xs = x.unsqueeze(0) if x.dim() == 1 else x.unsqueeze(1) # Ensure query_xs is (n_datapoints, input_dim)
        representations = theta_prev  # shape (n_regressors,)
        if type(self.model) == FunctionEncoder:  
            # FunctionEncoder expects query_xs with shape (batch_size, n_datapoints, input_dim)
            # and representations with shape (batch_size, n_regressors)
            query_xs = query_xs.unsqueeze(0)  # Add batch dimension
            representations = representations.unsqueeze(0)
            y_hat = self.model.predict(query_xs, representations=representations) # returns shape (batch_size, n_datapoints, output_dim)
            y_hat = y_hat.squeeze(0).squeeze(0)  # Remove batch and n_datapoints dimensions
        else:
            # For PolynomialModel, we can directly use the predict method
            y_hat = self.model.predict(query_xs, representations)
        #  Ensure y_hat is (output_dim,)
        if y_hat.dim() != 1 or y_hat.shape[0] != self.model.output_size[0]:
            raise ValueError(f"Invalid prediction shape: {y_hat.shape}")

        # 3. Compute the prediction error (innovation)
        e = y - y_hat # shape (output_dim,)
        if e.shape[-1] != self.model.output_size[0]:
            raise ValueError(f"Invalid prediction error shape: {e.shape}")

        # 4. Compute the Kalman gain K
        # First, Compute the innovation covariance: shape (output_dim, output_dim)
        S = Phi @ self.P @ Phi.T + self.R
        if S.dim() != 2 or S.shape[0] != self.model.output_size[0] or S.shape[1] != self.model.output_size[0]:
            raise ValueError(f"Invalid innovation covariance shape: {S.shape}")    
        K = torch.linalg.solve(S, Phi @ self.P.T).T  # expect shape (output_dim, n_regressors)
        if K.dim() != 2 or K.shape[0] != self.n or K.shape[1] != self.model.output_size[0]:
            raise ValueError(f"Invalid Kalman gain shape: {K.shape}")

        # 4. Update the parameter estimate (coefficients)
        self.theta = self.theta + K @ e # expect shape (n_regressors,)
        if self.theta.dim() != 1 or self.theta.shape[0] != self.n:
            raise ValueError(f"Invalid coefficients shape: {self.theta.shape}")

        # 5. Update the covariance matrix
        self.P = self.P - K @ Phi @ self.P # expect shape (n_regressors, n_regressors)
        if self.P.dim() != 2 or self.P.shape[0] != self.n or self.P.shape[1] != self.n:
            raise ValueError(f"Invalid covariance matrix shape: {self.P.shape}")

        # # Force symmetry and positive definiteness
        # self.P = (self.P + self.P.T) / 2  # Ensure symmetry
        
        if debug_mode:
            self.debug_info()

        info = {
            'theta_prev': theta_prev,
            'P_prev': P_prev,
            'Phi': Phi,
            'y_hat': y_hat,
            'e': e,
            'K': K,
            'P_new': self.P
        }

        return self.theta, info
    
class LMSStyleRLS(BaseRLS):
    """RLS in the LMS-style formulation, where the update is based on the gradient of the prediction error."""
    def __init__(self, n_regressors, **kwargs):
        super().__init__(n_regressors, **kwargs)

    def update(self, x, y, debug_mode=True):
        assert len(x.shape) in [1], f"Expected x to have shape (d,), got {x.shape}"
        assert len(y.shape) in [1], f"Expected y to have shape (m,), got {y.shape}"

        theta_prev = self.theta.clone()  # Store previous coefficients
        P_prev = self.P.clone() 

        # 1. Compute the regressor matrix (basis function representation of the input)
        Phi = self.model.forward_basis_functions(x) # shape (n_outputs, n_regressors)
        if Phi.dim() != 2 or Phi.shape[0] != self.model.output_size[0] or Phi.shape[1] != self.n:
            raise ValueError(f"Invalid regressor shape: {Phi.shape}")

        # 2. Compute the prediction y_hat = Phi @ theta_prev with shape (n_outputs,)
        query_xs = x.unsqueeze(0) if x.dim() == 1 else x.unsqueeze(1) # Ensure query_xs is (n_datapoints, input_dim)
        representations = theta_prev  # shape (n_regressors,)
        if type(self.model) == FunctionEncoder:  
            # FunctionEncoder expects query_xs with shape (batch_size, n_datapoints, input_dim)
            # and representations with shape (batch_size, n_regressors)
            query_xs = query_xs.unsqueeze(0)  # Add batch dimension
            representations = representations.unsqueeze(0)
            y_hat = self.model.predict(query_xs, representations=representations) # returns shape (batch_size, n_datapoints, output_dim)
            y_hat = y_hat.squeeze(0).squeeze(0)  # Remove batch and n_datapoints dimensions
        else:
            # For PolynomialModel, we can directly use the predict method
            y_hat = self.model.predict(query_xs, representations)
        #  Ensure y_hat is (output_dim,)
        if y_hat.dim() != 1 or y_hat.shape[0] != self.model.output_size[0]:
            raise ValueError(f"Invalid prediction shape: {y_hat.shape}")

        # 3. Compute the prediction error (innovation)
        e = y - y_hat # shape (output_dim,)
        if e.shape[-1] != self.model.output_size[0]:
            raise ValueError(f"Invalid prediction error shape: {e.shape}")

        # 4. Compute the Kalman gain K. LMS uses a scaled version of the identity matrix for the innovation covariance. Here we will incorporate a scaling factor based on the energy in the regressor matrix.
        # Note: the function encoder Gram matrix assumes the rows are linearly independent.
        gram = self.model._inner_product(Phi.unsqueeze(0).unsqueeze(0), Phi.unsqueeze(0).unsqueeze(0)).squeeze(0)
        assert gram.shape == (self.n, self.n), f"gram.shape: {gram.shape}" # expect shape (n_regressors, n_regressors)
        scaling_factor = 1 / (1 + (gram * self.P).sum())
        K = (self.P @ Phi.T) * scaling_factor
        if K.dim() != 2 or K.shape[0] != self.n or K.shape[1] != self.model.output_size[0]:
            raise ValueError(f"Invalid Kalman gain shape: {K.shape}")

        # 5. Update the parameter estimate (coefficients)
        self.theta = self.theta + K @ e # expect shape (n_regressors,)
        if self.theta.dim() != 1 or self.theta.shape[0] != self.n:
            raise ValueError(f"Invalid coefficients shape: {self.theta.shape}")

        # 6. Update the covariance matrix
        numerator = self.P @ gram @ self.P
        self.P = self.P - (numerator * scaling_factor) # expect shape (n_regressors, n_regressors)
        if self.P.dim() != 2 or self.P.shape[0] != self.n or self.P.shape[1] != self.n:
            raise ValueError(f"Invalid covariance matrix shape: {self.P.shape}")

        if debug_mode:
            self.debug_info()

        info = {
            'theta_prev': theta_prev,
            'P_prev': P_prev,
            'Phi': Phi,
            'y_hat': y_hat,
            'e': e,
            'K': K,
            'P_new': self.P
        }

        return self.theta, info

    
# class RegularizedRLS(BaseRLS):
#     """RLS with Tikhonov regularization γ on each update."""
#     def __init__(self, n_regressors, reg_param=1e-2, **kwargs):
#         super().__init__(n_regressors, **kwargs)
#         self.γ = reg_param

#     def update(self, phi, y):
#         # augment phi, y to absorb regularization
#         A = np.vstack([phi.T, np.sqrt(self.γ) * np.eye(self.n)])
#         b = np.vstack([y, np.zeros((self.n,1))])
#         # solve (A^T A) θ = A^T b via RLS step
#         K = self.P @ phi / (1 + phi.T @ self.P @ phi)
#         e = y - float(phi.T @ self.theta)
#         self.theta += K * e
#         self.P = self.P - K @ phi.T @ self.P


# class SiRLS(BaseRLS):
#     """RLS in the S_i formulation (store S_i = Φ_i^T Φ_i)."""
#     def __init__(self, n_regressors, Si0=None, **kwargs):
#         super().__init__(n_regressors, **kwargs)
#         # instead of P, track S = Φ^T Φ
#         self.S = Si0 if Si0 is not None else np.zeros((n_regressors, n_regressors))

#     def update(self, phi, y):
#         self.S += phi @ phi.T
#         # solve θ = S^{-1} * (Φ^T y) incrementally
#         # maintain z = Φ^T y
#         if not hasattr(self, 'z'):
#             self.z = np.zeros((self.n,1))
#         self.z += phi * y
#         self.theta = np.linalg.solve(self.S + 1e-8*np.eye(self.n), self.z)


# class KalmanRLS(BaseRLS):
#     """View RLS as a Kalman filter on θ with identity dynamics."""
#     def __init__(self, n_regressors, Q=None, R=1.0, **kwargs):
#         super().__init__(n_regressors, **kwargs)
#         # process noise (θ evolution) and measurement noise
#         self.Q = Q if Q is not None else np.eye(n_regressors)*1e-6
#         self.R = R

#     def update(self, phi, y):
#         # Prediction step (θ_k|k-1 = θ_{k-1}, P_k|k-1 = P_{k-1} + Q)
#         P_pred = self.P + self.Q

#         # Measurement update
#         S = phi.T @ P_pred @ phi + self.R
#         K = P_pred @ phi / float(S)
#         e = y - float(phi.T @ self.theta)

#         self.theta = self.theta + K * e
#         self.P = (np.eye(self.n) - K @ phi.T) @ P_pred


# Factory function to create RLS instances based on method name
def make_rls(method: str, **kwargs) -> BaseRLS:
    methods = {
        'standard': StandardRLS,
        'ff': ForgettingFactorRLS,
        'weighted': WeightedRLS,
        'lms-style': LMSStyleRLS,
        # 'reg': RegularizedRLS,
        # 'si': SiRLS,
        # 'kf': KalmanRLS,
    }
    if method not in methods:
        raise ValueError(f"Unknown RLS variant '{method}'")
    return methods[method](**kwargs)



# """
# Track diagnostics of the RLS update.
# - conditioning of the covariance matrix: condition number, positive definiteness
# - parameter error norm
# - prediction error norm
# - kalman gain norm
# - regressor: persistence of excitation
# """