
from os import popen
from select import POLLWRBAND
import tensorflow as tf
from tensorflow.keras import optimizers

def create_signed_pow(p):
  def signed_pow(x):
    return tf.sign(x)*tf.pow(tf.abs(x), p) 
  return signed_pow

class PowKernelGradientDescentOptimiser(optimizers.Optimizer):
  
  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.01,
               momentum=0.0,
               nesterov=False,
               pow = 1.0,
               name="SGD",
               **kwargs):
    super().__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("power", pow)
    self._set_hyper("decay", self._initial_decay)

    self._power = pow
    self.pow_function = create_signed_pow(pow)

    self._momentum = False
    if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError(f"`momentum` must be between [0, 1]. Received: "
                       f"momentum={momentum} (of type {type(momentum)}).")
    self._set_hyper("momentum", momentum)

    self.nesterov = nesterov

  def _create_slots(self, var_list):
    if self._momentum:
      for var in (var_list):
        self.add_slot(var, "momentum")

          
  def _prepare_local(self, var_device, var_dtype, apply_state):
    super()._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)]["momentum"] = tf.identity(
        self._get_hyper("momentum", var_dtype))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    if 'ernel' in var.name:
      grad = self.pow_function(grad)

    if self._momentum:
      momentum_var = self.get_slot(var, "momentum")
      return tf.raw_ops.ResourceApplyKerasMomentum(
          var=var.handle,
          accum=momentum_var.handle,
          lr=coefficients["lr_t"],
          grad=grad,
          momentum=coefficients["momentum"],
          use_locking=self._use_locking,
          use_nesterov=self.nesterov)
    else:
      return tf.raw_ops.ResourceApplyGradientDescent(
          var=var.handle,
          alpha=coefficients["lr_t"],
          delta=grad,
          use_locking=self._use_locking)

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices,
                                               **kwargs):
    
    
    if 'ernel' in var.name:
      grad = self.pow_function(grad)

    if self._momentum:
      return super()._resource_apply_sparse_duplicate_indices(
          grad, var, indices, **kwargs)
    else:
      var_device, var_dtype = var.device, var.dtype.base_dtype
      coefficients = (kwargs.get("apply_state", {}).get((var_device, var_dtype))
                      or self._fallback_apply_state(var_device, var_dtype))

      return tf.raw_ops.ResourceScatterAdd(
          resource=var.handle,
          indices=indices,
          updates=-grad * coefficients["lr_t"])

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # This method is only needed for momentum optimization.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    grad = self.pow_function(grad)

    momentum_var = self.get_slot(var, "momentum")
    return tf.raw_ops.ResourceSparseApplyKerasMomentum(
        var=var.handle,
        accum=momentum_var.handle,
        lr=coefficients["lr_t"],
        grad=grad,
        indices=indices,
        momentum=coefficients["momentum"],
        use_locking=self._use_locking,
        use_nesterov=self.nesterov)

  def get_config(self):
    config = super().get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "decay": self._initial_decay,
        "momentum": self._serialize_hyperparameter("momentum"),
        "nesterov": self.nesterov,
        "power": self._power
    })
    return config
