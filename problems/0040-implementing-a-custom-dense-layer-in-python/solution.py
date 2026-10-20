
import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure

class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        input_dim = self.input_shape[0]
        limit = 1 / np.sqrt(input_dim)

        self.w0 = np.zeros((1, self.n_units))
        self.W = np.random.uniform(
            -limit, limit, (input_dim, self.n_units)
        )

        self.w0_opt = copy.copy(optimizer)
        self.W_opt = copy.copy(optimizer)

    def parameters(self):
        return self.W.size + self.w0.size

    def forward_pass(self, X, training=True):
        self.layer_input = X
        y = X @ self.W + self.w0
        return y

    def backward_pass(self, accum_grad):
        # 必须先计算输入梯度，使用更新前的 W
        grad_input = accum_grad @ self.W.T

        if self.trainable:
            grad_W = self.layer_input.T @ accum_grad

            grad_w0 = np.sum(
                accum_grad, axis=0, keepdims=True
            )

            self.W = self.W_opt.update(self.W, grad_W)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        return grad_input

    def output_shape(self):
        return (self.n_units,)
