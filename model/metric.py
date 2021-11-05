### R2, RMSE, RMAE, CC

import numpy as np
import torch

def average_correlation_coefficient(y_pred, y_true):
    """Calculate Average Correlation Coefficient
    Args:
        y_true (array-like): np.ndarray or torch.Tensor of dimension N x d with actual values
        y_pred (array-like): np.ndarray or torch.Tensor of dimension N x d with predicted values
    Returns:
        float: Average Correlation Coefficient 
    Raises: 
        ValueError : If Parameters are not both of type np.ndarray or torch.Tensor 
    """
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        top = np.sum( (y_true - np.mean(y_true, axis=0)) *
                     (y_pred - np.mean(y_pred, axis=0) ), axis=0)
        bottom = np.sqrt( np.sum( (y_true - np.mean(y_true, axis=0))**2, axis=0 )
                         * np.sum((y_pred - np.mean(y_pred, axis=0))**2, axis=0) )
        return np.sum(top / bottom) / len(y_true[0])

    elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        top = torch.sum((y_true - torch.mean(y_true, dim=0))
                        * (y_pred - torch.mean(y_pred, dim=0)), dim=0)
        bottom = torch.sqrt(torch.sum((y_true - torch.mean(y_true, dim=0))**2, dim=0) *
                            torch.sum((y_pred - torch.mean(y_pred, dim=0))**2, dim=0))
        return torch.sum(top / bottom) / len(y_true[0])

    else:
        raise ValueError(
            'y_true and y_pred must be both of type numpy.ndarray or torch.Tensor')        
