from typing import Union, Tuple
import torch
from FunctionEncoder.Model.FunctionEncoder import FunctionEncoder

# Implements Weighted Recursive Regularized Least Squares (WRRLS) algorithm
class WRRLSFunctionEncoder(FunctionEncoder):
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
        
   
    def recursive_update(self, x, y, coefficients=None, Q_tilde=None):
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

        # 1. Compute the regressor (basis function representation of the input)
        psi = self.model.forward(x).T # shape: (n_basis,)

        # 2. Compute the prediction error (innovation)
        y_hat = psi.T @ theta_prev
        e = y - y_hat

        # 3. Compute the Kalman gain
        # 3a. Reconstruct and invert the regularized Gram matrix R
        alpha_I = self.alpha * torch.eye(self.n_basis, device='cuda')
        Q = Q_tilde_prev + alpha_I
        R = self.forgetting_factor * Q + (1 - self.forgetting_factor) * alpha_I
        P = torch.inverse(R)
        # 3b. Compute the Kalman gain
        K = (P @ psi) / (self.forgetting_factor + psi.T @ P @ psi)

        # 4. Update the parameter estimate (coefficients)
        theta_new = theta_prev + K * e

        # 5. Update data-dependent portion of the Gram matrix
        Q_tilde_new = self.forgetting_factor * Q_tilde_prev + psi @ psi.T

        # 6. Update internal state and return the updated coefficients and covariance matrix
        self.Q_tilde = Q_tilde_new
        coefficients = theta_new

        self.coefficients = coefficients

        info = {'theta_prev': theta_prev, 'Q_tilde_prev': Q_tilde_prev, 'psi': psi, 'y_hat': y_hat, 'e': e, 'K': K, 'Q_tilde_new': Q_tilde_new}

        return coefficients, info
    