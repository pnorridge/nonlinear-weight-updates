
"""GradientDescent for TensorFlow."""

import tensorflow as tf
import tensorflow_addons as tfa
from .PowKernelGradientDescent import PowKernelGradientDescentOptimiser
from tensorflow_addons.optimizers.weight_decay_optimizers import extend_with_decoupled_weight_decay

  
PowKernelGradientDescentOptimizerW = extend_with_decoupled_weight_decay(
    base_optimizer = PowGradientDescentOptimiser) 

class PowKernelGradientDescentOptimizerW_scaled(PowKernelGradientDescentOptimizerW):
    def __init__(self, learning_rate, weight_decay_opt, **kwargs):
        super().__init__( learning_rate = learning_rate, 
        weight_decay_opt = tf.multiply(tf.cast(weight_decay_opt, tf.float32),tf.cast(learning_rate, tf.float32)), 
        **kwargs)
  
  
class PowKernelGradientDescentOptimizerW_momscaled(PowKernelGradientDescentOptimizerW):
    def __init__(self, learning_rate, weight_decay_opt, momentum = 0.0, **kwargs):
        tmp = tf.math.reciprocal(tf.subtract(1., tf.cast(momentum, tf.float32)))
        mom_scale = tf.multiply(tmp, tf.cast(learning_rate, tf.float32))
        weight_decay_opt = tf.multiply(tf.cast(weight_decay_opt, tf.float32),mom_scale)
        super().__init__(learning_rate = learning_rate, 
                            weight_decay_opt = weight_decay_opt, 
                            **kwargs)
