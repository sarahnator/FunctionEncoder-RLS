import torch
import matplotlib.pyplot as plt

"""
rls.py

Several variants of Recursive Least Squares:
 - Plain RLS
 - Forgetting‐factor RLS
 - Regularized RLS
 - “S_i” formulation
 - Kalman‐filter extension
"""

import torch
from abc import ABC, abstractmethod
from FunctionEncoder.Model.FunctionEncoder import FunctionEncoder

class BaseRLS(ABC):
    """
    Abstract base class. Holds common state: theta (coefficients), P (covariance).
    """
    def __init__(self, n_regressors, fe_model, P0=None, theta0=None):
        self.n = n_regressors
        self.model = fe_model
        self.theta = theta0 if theta0 is not None else torch.zeros((n_regressors))
        # default P0: large diagonal to ensure initial trust in theta?
        self.P = P0 if P0 is not None else torch.eye(n_regressors) * 1e6

        # Add a dimension to P to match the batch size of theta
        if self.theta.dim() == 1:
            self.theta = self.theta.unsqueeze(0)
        if self.theta.dim() == 2 and self.P.dim() == 2:
            self.P = torch.stack([self.P] * self.theta.shape[0], dim=0)  # Expand P to match batch size of theta

        # move to device if model is on GPU
        if next(self.model.parameters()).is_cuda:
            self.theta = self.theta.cuda()
            self.P = self.P.cuda()

    @abstractmethod
    def update(self, x: torch.tensor, y: torch.tensor):
        """
        x: input vector (batch_dim, input_dim,) 
        y: vector measurement (batch_dim, output_dim,) 
        """
        pass

class StandardRLS(BaseRLS):
    """Plain RLS with optional regularization via initial P."""

    def update(self, x, y):
        assert len(x.shape) in [1,2], f"Expected x to have shape (b,d) or (d,), got {x.shape}"
        assert len(y.shape) in [1,2], f"Expected y to have shape (b,m) or (m,), got {y.shape}"

        theta_prev = self.theta.clone()  # Store previous coefficients
        P_prev = self.P.clone() 
              
        # Get the regressor (basis function representation of the input)
        Phi = self.model.forward_basis_functions(x) 
        
        # Ensure Phi is (batch_size, n_outputs, n_regressors)
        if Phi.dim() != 3 or Phi.shape[1] != self.model.output_size[0] or Phi.shape[-1] != self.n:
            raise ValueError(f"Invalid regressor shape: {Phi.shape}")

        #  Compute the prediction
        query_xs = x.unsqueeze(0) if x.dim() == 1 else x.unsqueeze(1) # Ensure query_xs is (batch_size, n_datapoints, input_dim)
        representations = theta_prev  # Ensure representations is (batch_dim, n_regressors)
        y_hat = self.model.predict(query_xs, representations=representations) # predicted shape (batch_size, n_datapoints, output_dim)
        
        # Ensure y_hat is (batch_size, output_dim)
        y_hat = y_hat.squeeze(1) # Remove n_points dimension

        #  Compute the prediction
        y_hat2 = torch.einsum('bmn,bn->bm', Phi, self.theta)
        # check y_hat2 and y_hat are the same?
        if not torch.allclose(y_hat, y_hat2, atol=1e-6):
            raise ValueError(f"Prediction mismatch: {y_hat.shape} vs {y_hat2.shape}")
        
        # Compute the prediction error (innovation)
        e = y - y_hat # (batch_size, output_dim)
        if e.shape[-1] != self.model.output_size[0]:
            raise ValueError(f"Invalid prediction error shape: {e.shape}")

        # Compute the innovation covariance
        # S = Phi @ self.P @ Phi.T + torch.eye(self.model.output_size) * 1e-6  # small regularization to avoid singularity
        # S = Phi @ self.P @ Phi.T + torch.eye(self.model.output_size[0])
        S = torch.einsum('bmn,bnn->bmn', Phi, self.P)
        S = torch.einsum('bmn,bni->bmi', S, Phi.transpose(-1, -2))  # S is now (batch_size, output_dim, output_dim)
        S = S + torch.stack([torch.eye(self.model.output_size[0], device=S.device)] * S.shape[0], dim=0) 
        # Check that S has dimensions (batch_size, output_dim, output_dim)
        if S.dim() != 3 or S.shape[0] != self.theta.shape[0] or S.shape[1] != self.model.output_size[0] or S.shape[2] != self.model.output_size[0]:
            raise ValueError(f"Invalid innovation covariance shape: {S.shape}")
        
        try:
            S = torch.linalg.inv(S)  # Invert S to compute Kalman gain
        except RuntimeError:
            raise ValueError("Innovation covariance S is not invertible.")

        # Compute the Kalman gain
        # K = self.P @ Phi.T @ S
        K = torch.einsum('bnn,bnm->bnm', self.P, Phi.transpose(-1, -2))  # K is now (batch_size, n_regressors, output_dim)
        K = torch.einsum('bnm,bmi->bni', K, S)  # K is now (batch_size, n_regressors, output_dim)
        # Check that K has dimensions (batch_size, n_regressors, output_dim)
        if K.dim() != 3 or K.shape[1] != self.n or K.shape[-1] != self.model.output_size[0]:
            raise ValueError(f"Invalid Kalman gain shape: {K.shape}")

        # Update the parameter estimate (coefficients)
        # self.theta = self.theta + K @ e
        self.theta = torch.einsum('bnm,bm->bn', K, e)  # self.theta is now (batch_size, n_regressors)
        # Check that theta has dimensions (batch_size, n_regressors)
        if self.theta.dim() != 2 or self.theta.shape[0] != theta_prev.shape[0] or self.theta.shape[-1] != self.n:
            raise ValueError(f"Invalid coefficients shape: {self.theta.shape}")

        # Update the covariance matrix
        # self.P = self.P - K @ Phi @ self.P
        rank_k_update = torch.einsum('bmn,bni->bmi', Phi, self.P)  # rank-k update
        normalized_rank_k_update = torch.einsum('bnm,bmi->bni', K, rank_k_update)  # K @ Phi.T
        self.P = self.P - normalized_rank_k_update

        # Check that P has dimensions (batch_size, n_regressors, n_regressors)
        if self.P.dim() != 3 or self.P.shape[0] != self.theta.shape[0] or self.P.shape[1] != self.n or self.P.shape[2] != self.n:
            raise ValueError(f"Invalid covariance matrix shape: {self.P.shape}")
        
        # # Is P positive definite?
        # if not torch.all(torch.linalg.eigvals(self.P[0,...]).real > 0):
        #     raise ValueError("Covariance matrix P is not positive definite.")
        # # Check that P is symmetric
        # if not torch.allclose(self.P[0,...], self.P[0,...].T, atol=1e-6):
        #     raise ValueError("Covariance matrix P is not symmetric.")
        # # Check that P is well-conditioned
        # if torch.linalg.cond(self.P[0,...]) > 1e10:  # arbitrary threshold for conditioning
        #     raise ValueError("Covariance matrix P is poorly conditioned.")
        # # Check that P is not singular
        # if torch.linalg.det(self.P[0,...]) < 1e-10:  # arbitrary threshold for singularity
        #     raise ValueError("Covariance matrix P is singular or nearly singular.")
        # # Check that P is not too large
        # if torch.max(self.P[0,...]) > 1e10:  # arbitrary threshold for large values
        #     raise ValueError("Covariance matrix P has excessively large values.")
        # # Check that P is not too small
        # if torch.min(self.P[0,...]) < 1e-10:  # arbitrary threshold for small values
        #     raise ValueError("Covariance matrix P has excessively small values.")

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




# class ForgettingFactorRLS(BaseRLS):
#     """Forgetting‐factor RLS (decay parameter λ)."""
#     def __init__(self, n_regressors, forgetting_factor=0.99, **kwargs):
#         super().__init__(n_regressors, **kwargs)
#         self.λ = forgetting_factor

#     def update(self, phi, y):
#         Pφ = self.P @ phi
#         denom = self.λ + float(phi.T @ Pφ)
#         K = Pφ / denom
#         e = y - float(phi.T @ self.theta)
#         self.theta += K * e
#         self.P = (self.P - K @ phi.T @ self.P) / self.λ


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


# Factory if you like
def make_rls(method: str, **kwargs) -> BaseRLS:
    methods = {
        'standard': StandardRLS,
        # 'ff': ForgettingFactorRLS,
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