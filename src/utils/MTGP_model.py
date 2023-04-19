    # -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:16:27 2022

@author: aarregui
"""


import gpytorch



class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        
        tasks_num = train_y.shape[-1]
        
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=tasks_num
        )
        
        #Define kernel constraints
        length_constraint = gpytorch.constraints.Interval(1e-20, 1)
        
        
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(lengthscale_constraint=length_constraint),
            num_tasks=tasks_num, rank=0
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)