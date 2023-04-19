    # -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:16:27 2022

@author: aarregui
"""

import torch
import gpytorch
import numpy as np


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, tasks_num, latents_num, inducing_points):
                
        # Let's use a different set of inducing points for each latent function
        # We'll use the same set of inducing points for each latent function
        # torch.from_numpy()
        #generate inducing_point np.array, of shape (latents_num, n_inducing_points, 1)
        
        n_inducing_points = len(inducing_points)
        #inducing_points_array = np.repeat(inducing_points,latents_num).reshape((latents_num,n_inducing_points,1), order = 'F')
        inducing_points_array = inducing_points.reshape((latents_num, n_inducing_points//latents_num, 1), order = 'F')
        inducing_points = torch.from_numpy(inducing_points_array)
        
        
        
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([latents_num])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=False
            ),
            num_tasks=tasks_num,
            num_latents=latents_num,
            latent_dim=-1
        )

        super().__init__(variational_strategy)


        #Define kernel constraints
        length_constraint = gpytorch.constraints.Interval(1e-5, 10)
        
        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([latents_num]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_constraint=length_constraint,batch_shape=torch.Size([latents_num]))
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# model = MultitaskGPModel()
# likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=tasks_num)