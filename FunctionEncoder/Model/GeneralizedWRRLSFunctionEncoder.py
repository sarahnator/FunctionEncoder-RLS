from typing import Union, Tuple
import torch
from FunctionEncoder.Model.FunctionEncoder import FunctionEncoder

# Implements Weighted Recursive Regularized Least Squares (WRRLS) algorithm
class GeneralizedWRRLSFunctionEncoder(FunctionEncoder):
    def __init__(
        self,
        input_size:tuple[int], 
        output_size:tuple[int], 
        data_type:str, 
        n_basis:int=100, 
        model_type:Union[str, type]="MLP",
        model_kwargs:dict=dict(),
        method:str="least_squares", 
        use_residuals_method:bool=False,  
        regularization_parameter:float=1.0, # if you normalize your data, this is usually good
        gradient_accumulation:int=1, # default: no gradient accumulation
        optimizer=torch.optim.Adam,
        optimizer_kwargs:dict={"lr":1e-3},
        forgetting_factor:float=1, # set < 1 for time-varying parameters (typically between 0.98 and 0.995) and equal to 1 for time-invariant (constant) parameters -- https://www.mathworks.com/help/ident/ug/algorithms-for-online-estimation.html
        init_coefficients:torch.Tensor=None, # initial coefficients
        alpha:float=1e-3, # regularization parameter for the covariance matrix. Typically set to a small value (e.g., 1e-3) to ensure numerical stability. 
    ):
        
        self.forgetting_factor = forgetting_factor
        self.alpha = alpha
        self.Q_tilde = torch.zeros(n_basis, device='cuda') # this is the data-dependent part of the Gram matrix
        self.coefficients = init_coefficients

        super().__init__(
        input_size=input_size, 
        output_size=output_size, 
        data_type=data_type, 
        n_basis=n_basis, 
        model_type=model_type,
        model_kwargs=model_kwargs,
        method=method, 
        use_residuals_method=use_residuals_method,  
        regularization_parameter=regularization_parameter, 
        gradient_accumulation=gradient_accumulation,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs
    )
    
    def batch_recursive_update(self, x, y, coefficients=None, Q_tilde=None):
        """
        Perform a recursive least squares (RLS) update of the basis function coefficients with a regularization term.
        Update equations modified the forgetting factor adaptation algorithm (https://www.mathworks.com/help/ident/ug/algorithms-for-online-estimation.html)
        to include a regularization term.
        
        Notation is as follows:
        - x: input data
        - y: target data
        - coefficients: current coefficients
        - Q_tilde: data-dependent portion of the Gram (optional)
        - alpha: regularization parameter
        - theta_prev: previous coefficients
        - Q_tilde_prev: previous data-dependent portion of the Gram
        - psi: regressor (basis function representation of the input)
        - y_hat: predicted output
        - e: prediction error (innovation)
        - K: Kalman gain
        - theta_new: updated coefficients
        - Q_tilde_new: updated data-dependent portion of the Gram
        - R: regularized Gram matrix
        - P: covariance matrix (inverse of R)

        Args:
            x: input data
            y: target data
            coefficients: current coefficients
            Q_tilde: covariance matrix (optional)
        Returns:
            coefficients: updated coefficients
            info: additional information (optional)
        """
        theta_prev = coefficients if coefficients is not None else self.coefficients
        Q_tilde_prev = Q_tilde if Q_tilde is not None else self.Q_tilde

        # size check
        batch_size = x.shape[0]
        assert y.shape[0] == batch_size, f"y.shape: {y.shape}, x.shape: {x.shape}"
        assert theta_prev.shape == (batch_size, self.n_basis, 1), f"theta_prev.shape: {theta_prev.shape}, n_basis: {self.n_basis}"
        assert Q_tilde_prev.shape == (batch_size, self.n_basis, self.n_basis), f"Q_tilde_prev.shape: {Q_tilde_prev.shape}, n_basis: {self.n_basis}"

        # 1. Compute the regressor (basis function representation of the input)
        psi = self.model.forward(x).T # shape: (n_basis, n_output, batch_size)
        assert psi.shape == (self.n_basis, self.output_size[0], batch_size), f"psi.shape: {psi.shape}, n_basis: {self.n_basis}, output_size: {self.output_size}"

        # 2. Compute the prediction error (innovation)
        # assuming psi.T has shape (batch, 2, n_basis), theta_prev has shape (n_batch, n_basis, 1)
        # do batch matrix multiplication with einsum to align dimensions "b2k,bk1->b21"
        # y_hat = psi.T @ theta_prev
        y_hat = torch.einsum("bok,bki->boi", psi.T, theta_prev)
        y = y.reshape(y_hat.shape)
        assert y_hat.shape == y.shape, f"y_hat.shape: {y_hat.shape}, y.shape: {y.shape}"
        e = y - y_hat

        # 3. Compute the Kalman gain
        # 3a. Reconstruct and invert the regularized Gram matrix R
        alpha_I = self.alpha * torch.eye(self.n_basis, device='cuda')
        Q = Q_tilde_prev + alpha_I # broadcast to (batch_size, n_basis, n_basis)
        R = self.forgetting_factor * Q + (1 - self.forgetting_factor) * alpha_I
        P = torch.inverse(R)
        # 3b. Compute the Kalman gain
        # inner product expects inputs of shape (batch_size. n_datapoints, output_size, n_basis)
        gram = self._inner_product(psi.T.unsqueeze(1), psi.T.unsqueeze(1)).squeeze(0)
        assert gram.shape == (batch_size, self.n_basis, self.n_basis,), f"gram.shape: {gram.shape}"
        
        # the Kalman gain matrix computes
        # K = (P @ psi) / (self.forgetting_factor + (gram * P).sum())
        unnormalized_K = torch.einsum("bkk,kob->bko", P, psi)  # shape: (batch_size, n_basis, n_output)
        K = (unnormalized_K) / (self.forgetting_factor + (gram * P).sum()) # normalize the gain to scale the update relative to the prior uncertainty and new information


        # 4. Update the parameter estimate (coefficients)
        theta_new = theta_prev + K @ e

        # 5. Update data-dependent portion of the Gram matrix
        Q_tilde_new = self.forgetting_factor * Q_tilde_prev + gram

        # 6. Update internal state and return the updated coefficients and covariance matrix
        self.Q_tilde = Q_tilde_new
        coefficients = theta_new

        self.coefficients = coefficients
        assert self.coefficients.shape == (batch_size, self.n_basis, 1), f"self.coefficients.shape: {self.coefficients.shape}"

        info = {'theta_prev': theta_prev, 'Q_tilde_prev': Q_tilde_prev, 'psi': psi, 'y_hat': y_hat, 'e': e, 'K': K, 'Q_tilde_new': Q_tilde_new}

        return coefficients, info
    
   
    def recursive_update(self, x, y, coefficients=None, Q_tilde=None):
        """
        Batched recursive least squares (RLS) update.
        
        Args:
            x: input data of shape (B, input_dim)
            y: target data of shape (B,) or (B, 1)
            coefficients: current coefficients, shape (B, n_basis, 1)
            Q_tilde: data-dependent portion of the Gram matrix, shape (B, n_basis, n_basis)
            
        Returns:
            coefficients: updated coefficients, shape (B, n_basis, 1)
            info: additional information (optional)
        """
        # Use batched coefficients / covariance, or fallback to device-level defaults.
        # (These should be pre-initialized in a batched fashion.)
        theta_prev = coefficients if coefficients is not None else self.coefficients  # (B, n_basis, 1)
        Q_tilde_prev = Q_tilde if Q_tilde is not None else self.Q_tilde           # (B, n_basis, n_basis)
        
        batch_size = x.shape[0]
        
        # 1. Compute the regressor (basis function representation) for each batch sample.
        # Assume self.model.forward(x) returns shape (B, n_basis, n_output)
        psi = self.model.forward(x)  # shape: (B, n_basis, n_output)
        # If you need to work with psi transposed per sample (n_output first), use:
        # psi_T = psi.transpose(1, 2)  # shape: (B, n_output, n_basis)
        
        # 2. Compute the predicted output and the prediction error.
        # Here we compute y_hat = psi^T @ theta_prev per batch.
        # For each batch sample, psi[b] has shape (n_basis, n_output) so psi[b].T is (n_output, n_basis)
        # and theta_prev[b] is (n_basis, 1) so the bmm gives y_hat[b] with shape (n_output, 1).
        y_hat = torch.bmm(psi.transpose(1, 2), theta_prev)  # (B, n_output, 1)
        
        # Ensure y has shape (B, n_output, 1) – for example if n_output==1.
        y = y.view(batch_size, -1, 1)
        assert y_hat.shape == y.shape, f"y_hat.shape: {y_hat.shape}, y.shape: {y.shape}"
        e = y - y_hat  # (B, n_output, 1)
        
        # 3. Compute the Kalman gain
        # 3a. Build a batched alpha_I to add to Q_tilde_prev.
        # alpha_I has shape (n_basis, n_basis); we expand to (B, n_basis, n_basis)
        alpha_I = self.alpha * torch.eye(self.n_basis, device=x.device)
        alpha_I = alpha_I.unsqueeze(0).expand(batch_size, -1, -1)
        
        Q = Q_tilde_prev + alpha_I  # (B, n_basis, n_basis)
        R = self.forgetting_factor * Q + (1 - self.forgetting_factor) * alpha_I  # (B, n_basis, n_basis)
        P = torch.inverse(R)  # (B, n_basis, n_basis)
        
        # 3b. Compute the Gram matrix for each sample.
        # For each b, gram[b] = psi[b].T @ psi[b], yielding shape (n_basis, n_basis)
        gram = torch.bmm(psi.transpose(1, 2), psi)  # (B, n_basis, n_basis)
        
        # Denom: compute a scalar per batch sample.
        # Here, (gram * P) produces (B, n_basis, n_basis); summing over the last two dims gives (B,)
        denom = self.forgetting_factor + (gram * P).sum(dim=(-2, -1))  # (B,)
        denom = denom.view(batch_size, 1, 1)  # make it broadcastable
        
        # Compute Kalman gain: K = (P @ psi) / denom, where:
        # P @ psi: (B, n_basis, n_output) and after division, K has same shape.
        K = torch.bmm(P, psi) / denom  # (B, n_basis, n_output)
        
        # 4. Update the parameter estimate.
        # We want to add the correction: K @ e, where e has shape (B, n_output, 1)
        # so correction is (B, n_basis, 1) and theta_new remains (B, n_basis, 1).
        theta_new = theta_prev + torch.bmm(K, e)  # (B, n_basis, 1)
        
        # 5. Update the data-dependent portion of the Gram matrix.
        Q_tilde_new = self.forgetting_factor * Q_tilde_prev + gram  # (B, n_basis, n_basis)
        
        # 6. Update internal state and return info.
        self.Q_tilde = Q_tilde_new  # now batched
        self.coefficients = theta_new
        assert self.coefficients.shape == (batch_size, self.n_basis, 1), f"self.coefficients.shape: {self.coefficients.shape}"
        
        info = {
            'theta_prev': theta_prev,
            'Q_tilde_prev': Q_tilde_prev,
            'psi': psi,
            'y_hat': y_hat,
            'e': e,
            'K': K,
            'Q_tilde_new': Q_tilde_new
        }
        
        return theta_new, info

    # def recursive_update(self, x, y, coefficients=None, Q_tilde=None):
    #     """
    #     Perform a recursive least squares (RLS) update of the basis function coefficients with a regularization term.
    #     Update equations modified the forgetting factor adaptation algorithm (https://www.mathworks.com/help/ident/ug/algorithms-for-online-estimation.html)
    #     to include a regularization term.
        
    #     Notation is as follows:
    #     - x: input data
    #     - y: target data
    #     - coefficients: current coefficients
    #     - Q_tilde: data-dependent portion of the Gram (optional)
    #     - alpha: regularization parameter
    #     - theta_prev: previous coefficients
    #     - Q_tilde_prev: previous data-dependent portion of the Gram
    #     - psi: regressor (basis function representation of the input)
    #     - y_hat: predicted output
    #     - e: prediction error (innovation)
    #     - K: Kalman gain
    #     - theta_new: updated coefficients
    #     - Q_tilde_new: updated data-dependent portion of the Gram
    #     - R: regularized Gram matrix
    #     - P: covariance matrix (inverse of R)

    #     Args:
    #         x: input data
    #         y: target data
    #         coefficients: current coefficients
    #         Q_tilde: covariance matrix (optional)
    #     Returns:
    #         coefficients: updated coefficients
    #         info: additional information (optional)
    #     """
    #     theta_prev = coefficients if coefficients is not None else self.coefficients
    #     Q_tilde_prev = Q_tilde if Q_tilde is not None else self.Q_tilde

    #     # 1. Compute the regressor (basis function representation of the input)
    #     psi = self.model.forward(x).T # shape: (n_basis, n_output)

    #     # 2. Compute the prediction error (innovation)
    #     y_hat = psi.T @ theta_prev
    #     # e = y - y_hat        
    #     y = y.view(-1, 1)
    #     assert y_hat.shape == y.shape, f"y_hat.shape: {y_hat.shape}, y.shape: {y.shape}"
    #     e = y - y_hat

    #     # 3. Compute the Kalman gain
    #     # 3a. Reconstruct and invert the regularized Gram matrix R
    #     alpha_I = self.alpha * torch.eye(self.n_basis, device='cuda')
    #     Q = Q_tilde_prev + alpha_I
    #     R = self.forgetting_factor * Q + (1 - self.forgetting_factor) * alpha_I
    #     P = torch.inverse(R)
    #     # 3b. Compute the Kalman gain
    #     gram = self._inner_product(psi.T.unsqueeze(0).unsqueeze(0), psi.T.unsqueeze(0).unsqueeze(0)).squeeze(0)
    #     assert gram.shape == (self.n_basis, self.n_basis,), f"gram.shape: {gram.shape}"
    #     K = (P @ psi) / (self.forgetting_factor + (gram * P).sum())

    #     # 4. Update the parameter estimate (coefficients)
    #     theta_new = theta_prev + K @ e

    #     # 5. Update data-dependent portion of the Gram matrix
    #     Q_tilde_new = self.forgetting_factor * Q_tilde_prev + gram

    #     # 6. Update internal state and return the updated coefficients and covariance matrix
    #     self.Q_tilde = Q_tilde_new
    #     coefficients = theta_new

    #     self.coefficients = coefficients
    #     assert self.coefficients.shape == (self.n_basis, 1), f"self.coefficients.shape: {self.coefficients.shape}"

    #     info = {'theta_prev': theta_prev, 'Q_tilde_prev': Q_tilde_prev, 'psi': psi, 'y_hat': y_hat, 'e': e, 'K': K, 'Q_tilde_new': Q_tilde_new}

    #     return coefficients, info
    