from typing import Union, Tuple
import torch
from FunctionEncoder.Model.FunctionEncoder import FunctionEncoder

# Implements Weighted Recursive Regularized Least Squares (WRRLS) algorithm
class GeneralizedWRLSFunctionEncoder(FunctionEncoder):
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
        delta:float=1e-3, # regularization parameter for the covariance matrix. Typically set to a small value (e.g., 1e-3) to ensure numerical stability. This is the initial value of the covariance matrix P. Smaller noise variance suggests using a larger value for delta (implying lower initially uncertainty), whereas larger noise variance suggests using a smaller value for delta (implying higher initial uncertainty).
        init_coefficients:torch.Tensor=None, # initial coefficients
    ):
        
        self.forgetting_factor = forgetting_factor
        self.delta = delta
        self.P = torch.eye(n_basis, device='cuda') * (1 / self.delta)
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
        
   
    def recursive_update(self, x, y, coefficients=None, P=None):
        """
        Perform a recursive least squares (RLS) update of the basis function coefficients.
        Update equations are based on the forgetting factor adaptation algorithm: https://www.mathworks.com/help/ident/ug/algorithms-for-online-estimation.html
        
        Notation is as follows:
        - x: input data
        - y: target data
        - coefficients: current coefficients
        - P: covariance matrix (optional)
        - theta_prev: previous coefficients
        - P_prev: previous covariance matrix
        - psi: regressor (basis function representation of the input)
        - y_hat: predicted output
        - e: prediction error (innovation)
        - K: Kalman gain
        - theta_new: updated coefficients
        - P_new: updated covariance matrix

        Args:
            x: input data
            y: target data
            coefficients: current coefficients
            P: covariance matrix (optional)
        Returns:
            coefficients: updated coefficients
            info: additional information (optional)
        """
        theta_prev = coefficients if coefficients is not None else self.coefficients
        P_prev = P if P is not None else self.P

        # 1. Compute the regressor (basis function representation of the input)
        psi = self.model.forward(x).T # shape: (n_basis, n_output)

        # 2. Compute the prediction error (innovation)
        y_hat = psi.T @ theta_prev
        y = y.view(-1, 1)
        assert y_hat.shape == y.shape, f"y_hat.shape: {y_hat.shape}, y.shape: {y.shape}"
        e = y - y_hat

        # 3. Compute the Kalman gain 
        gram = self._inner_product(psi.T.unsqueeze(0).unsqueeze(0), psi.T.unsqueeze(0).unsqueeze(0)).squeeze(0)
        assert gram.shape == (self.n_basis, self.n_basis,), f"gram.shape: {gram.shape}"
        K = (P_prev @ psi) / (self.forgetting_factor + (gram * P_prev).sum())

        # 4. Update the parameter estimate (coefficients)
        theta_new = theta_prev + K @ e

        # 5. Update the regularization parameter (covariance matrix)
        numerator = P_prev @ gram @ P_prev  
        denominator = (self.forgetting_factor + (gram * P_prev).sum())
        P_new = (1 / self.forgetting_factor) * (P_prev - (numerator / denominator))

        # 6. Update internal state and return the updated coefficients and covariance matrix
        self.P = P_new
        coefficients = theta_new

        self.coefficients = coefficients
        assert self.coefficients.shape == (self.n_basis, 1), f"self.coefficients.shape: {self.coefficients.shape}"

        info = {'theta_prev': theta_prev, 'P_prev': P_prev, 'psi': psi, 'y_hat': y_hat, 'e': e, 'K': K, 'P_new': P_new}

        return coefficients, info
    