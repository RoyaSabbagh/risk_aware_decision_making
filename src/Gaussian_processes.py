#!/usr/bin/env python
"""
Gaussian_processes.py

Implementation of gaussian processes models for human motion and intention
learning and prediction.

Author: Roya Sabbagh Novin (sabbaghnovin@gmail.com)
"""

import numpy as np
from scipy import *
import random
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import gpflow
from gpflow.utilities import to_default_float
float_type = gpflow.config.default_float()


def randomize(model, mean=1, sigma=0.01):
    model.kernel.lengthscales.assign(
        mean + sigma*np.random.normal(size=model.kernel.lengthscales.shape))
    model.kernel.variance.assign(
        mean + sigma*np.random.normal(size=model.kernel.variance.shape))
    if model.likelihood.variance.trainable:
        model.likelihood.variance.assign(
            mean + sigma*np.random.normal())

class MGPR(gpflow.Module):
    """
    class definition for GP models.
    """
    def __init__(self, data, name=None):
        """
        Initialization:
        The dimentions are extracted from the data.
        """
        super(MGPR, self).__init__(name)

        self.num_outputs = data[1].shape[1]
        self.num_dims = data[0].shape[1]
        self.num_datapoints = data[0].shape[0]

        self.create_models(data)
        self.optimizers = []

    def create_models(self, data):
        """
        Craet model using squared exponential kernel.
        """
        self.models = []
        for i in range(self.num_outputs):
            kern = gpflow.kernels.SquaredExponential(lengthscales=0.1*tf.ones([data[0].shape[1],], dtype=float_type))
            self.models.append(gpflow.models.GPR((data[0], data[1][:, i:i+1]), kernel=kern))

    def save_model(self):
        """
        Finding parameters for model saving.
        """
        params_set = []
        for model in self.models:
            params = gpflow.utilities.parameter_dict(model)
            gpflow.utilities.reset_cache_bijectors(model)
            params_set.append(params)
        return params_set

    def optimize(self, restarts=1):
        """
        Find model paramters through optimization.
        """
        if len(self.optimizers) == 0:  # This is the first call to optimize();
            for model in self.models:
                # Create an gpflow.train.ScipyOptimizer object for every model embedded in mgpr
                optimizer = gpflow.optimizers.Scipy()
                optimizer.minimize(model.training_loss, model.trainable_variables)
                self.optimizers.append(optimizer)
        else:
            for model, optimizer in zip(self.models, self.optimizers):
                optimizer.minimize(model.training_loss, model.trainable_variables)

        for model, optimizer in zip(self.models, self.optimizers):
            best_params = {
                "lengthscales" : model.kernel.lengthscales.value(),
                "k_variance" : model.kernel.variance.value(),
                "l_variance" : model.likelihood.variance.value()}
            best_loss = model.training_loss()
            for restart in range(restarts):
                randomize(model)
                optimizer.minimize(model.training_loss, model.trainable_variables)
                loss = model.training_loss()
                if loss < best_loss:
                    best_params["k_lengthscales"] = model.kernel.lengthscales.value()
                    best_params["k_variance"] = model.kernel.variance.value()
                    best_params["l_variance"] = model.likelihood.variance.value()
                    best_loss = model.training_loss()
            model.kernel.lengthscales.assign(best_params["lengthscales"])
            model.kernel.variance.assign(best_params["k_variance"])
            model.likelihood.variance.assign(best_params["l_variance"])

    def predict_intention_probability(self, traj, prob):
        """
        Find the probability of intentions.
        IN: traj = obsereved trajectory
            prob = prior on intentions
        OUT: goal_probability = probablity of a given intention
        """
        goal_probability = prob
        lam = 0.1 # forgettting factor
        for j in range(len(traj)-1):
            M, S, _, _ = self.propagate(traj[j][0][0:2],  0.01 * np.eye(2))
            diff = traj[j+1][0][0:2] - M.numpy()[0]
            p_next=np.power(np.linalg.det(np.multiply(2*np.pi,S.numpy())),-0.5)*np.exp(-0.5*np.dot(np.dot(np.transpose(diff),np.linalg.inv(S.numpy())), diff))
            goal_probability = p_next * np.power(goal_probability, (1-lam))
        return goal_probability

    def predict_path(self, start, n):
        """
        Predict the next n steps based on the model given a initial state.
        IN: start = initial state
            n = length of horizon
        OUT: x, dx = path, differences in state
        """
        dx = [[0,0]]
        x=[start[0][0:2]]

        for j in range(n-1):
            _, _, m_dx, s_dx = self.propagate(x[j], 0 * np.eye(2))
            dx.append([np.random.normal(m_dx.numpy()[0][0], s_dx.numpy()[0][0]), np.random.normal(m_dx.numpy()[0][1], s_dx.numpy()[1][1])])
            x.append(x[j] + dx[j+1])
        return [x, dx]

    def predict(self, m_x, s_x, n):
        prediction = []
        for j in range(n):
            m_x, s_x, m_dx, s_dx = self.propagate(m_x, s_x)
            prediction.append([m_x.numpy()[0], s_x.numpy()])
        return prediction

    def propagate(self, m_x, s_x):
        M_dx, S_dx, C_dx = self.predict_on_noisy_inputs(m_x, s_x)
        M_x = M_dx + m_x
        S_x = S_dx + s_x

        # While-loop requires the shapes of the outputs to be fixed
        M_x.set_shape([1, self.num_outputs]); S_x.set_shape([self.num_outputs, self.num_outputs])
        M_dx.set_shape([1, self.num_outputs]); S_dx.set_shape([self.num_outputs, self.num_outputs])
        return M_x, S_x, M_dx, S_dx

    def predict_on_noisy_inputs(self, m, s):
        iK, beta = self.calculate_factorizations()
        return self.predict_given_factorizations(m, s, iK, beta)

    def calculate_factorizations(self):
        K = self.K(self.X)
        batched_eye = tf.eye(tf.shape(self.X)[0], batch_shape=[self.num_outputs], dtype=float_type)
        L = tf.linalg.cholesky(K + self.noise[:, None, None]*batched_eye)
        iK = tf.linalg.cholesky_solve(L, batched_eye, name='chol1_calc_fact')
        Y_ = tf.transpose(self.Y)[:, :, None]
        beta = tf.linalg.cholesky_solve(L, Y_, name="chol2_calc_fact")[:, :, 0]
        return iK, beta

    def predict_given_factorizations(self, m, s, iK, beta):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """

        s = tf.tile(s[None, None, :, :], [self.num_outputs, self.num_outputs, 1, 1])
        inp = tf.tile(self.centralized_input(m)[None, :, :], [self.num_outputs, 1, 1])

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = tf.linalg.diag(1/self.lengthscales)
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + tf.eye(self.num_dims, dtype=float_type)

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t = tf.linalg.matrix_transpose(
                tf.linalg.solve(B, tf.linalg.matrix_transpose(iN), adjoint=True, name='predict_gf_t_calc'),
            )

        lb = tf.exp(-tf.reduce_sum(iN * t, -1)/2) * beta
        tiL = t @ iL
        c = self.variance / tf.sqrt(tf.linalg.det(B))

        M = (tf.reduce_sum(lb, -1) * c)[:, None]
        V = tf.matmul(tiL, lb[:, :, None], adjoint_a=True)[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        R = s @ tf.linalg.diag(
                1/tf.square(self.lengthscales[None, :, :]) +
                1/tf.square(self.lengthscales[:, None, :])
            ) + tf.eye(self.num_dims, dtype=float_type)

        X = inp[None, :, :, :]/tf.square(self.lengthscales[:, None, None, :])
        X2 = -inp[:, None, :, :]/tf.square(self.lengthscales[None, :, None, :])
        Q = tf.linalg.solve(R, s, name='Q_solve')/2
        Xs = tf.reduce_sum(X @ Q * X, -1)
        X2s = tf.reduce_sum(X2 @ Q * X2, -1)
        maha = -2 * tf.matmul(X @ Q, X2, adjoint_b=True) + \
            Xs[:, :, :, None] + X2s[:, :, None, :]

        k = tf.math.log(self.variance)[:, None] - \
            tf.reduce_sum(tf.square(iN), -1)/2
        L = tf.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S = (tf.tile(beta[:, None, None, :], [1, self.num_outputs, 1, 1])
                @ L @
                tf.tile(beta[None, :, :, None], [self.num_outputs, 1, 1, 1])
            )[:, :, 0, 0]

        diagL = tf.transpose(tf.linalg.diag_part(tf.transpose(L)))
        S = S - tf.linalg.diag(tf.reduce_sum(tf.multiply(iK, diagL), [1, 2]))
        S = S / tf.sqrt(tf.linalg.det(R))
        S = S + tf.linalg.diag(self.variance)
        S = S - M @ tf.transpose(M)

        return tf.transpose(M), S, tf.transpose(V)

    def centralized_input(self, m):
        return self.X - m

    def K(self, X1, X2=None):
        return tf.stack(
            [model.kernel.K(X1, X2) for model in self.models]
        )

    @property
    def Y(self):
        return tf.concat(
            [model.data[1] for model in self.models],
            axis = 1
        )

    @property
    def X(self):
        return self.models[0].data[0]

    @property
    def lengthscales(self):
        return tf.stack(
            [model.kernel.lengthscales.value() for model in self.models]
        )

    @property
    def variance(self):
        return tf.stack(
            [model.kernel.variance.value() for model in self.models]
        )

    @property
    def noise(self):
        return tf.stack(
            [model.likelihood.variance.value() for model in self.models]
        )

    @property
    def data(self):
        return (self.X, self.Y)
