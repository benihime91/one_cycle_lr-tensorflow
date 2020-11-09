from tensorflow import keras
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import numpy as np

K = keras.backend


class OneCycleLr(keras.callbacks.Callback):
    """
    Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    [Implementation taken from PyTorch:
    (https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR)]

    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch
    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    Args:
    max_lr (float): Upper learning rate boundaries in the cycle.
    total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
    epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
    steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
    pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
    anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
            linear annealing.
            Default: 'cos'
    cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
    base_momentum (float): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.85
    max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.95
    div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
    final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
    """

    def __init__(self,
                 max_lr: float,
                 total_steps: int = None,
                 epochs: int = None,
                 steps_per_epoch: int = None,
                 pct_start: float = 0.3,
                 anneal_strategy: str = "cos",
                 cycle_momentum: bool = True,
                 base_momentum: float = 0.85,
                 max_momentum: float = 0.95,
                 div_factor: float = 25.0,
                 final_div_factor: float = 1e4,
                 ) -> None:

        super(OneCycleLr, self).__init__()

        # validate total steps:
        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError(
                "You must define either total_steps OR (epochs AND steps_per_epoch)"
            )
        elif total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError(
                    "Expected non-negative integer total_steps, but got {}".format(
                        total_steps
                    )
                )
            self.total_steps = total_steps
        else:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError(
                    "Expected non-negative integer epochs, but got {}".format(
                        epochs)
                )
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError(
                    "Expected non-negative integer steps_per_epoch, but got {}".format(
                        steps_per_epoch
                    )
                )
            # Compute total steps
            self.total_steps = epochs * steps_per_epoch

        self.step_num = 0
        self.step_size_up = float(pct_start * self.total_steps) - 1
        self.step_size_down = float(self.total_steps - self.step_size_up) - 1

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError(
                "Expected float between 0 and 1 pct_start, but got {}".format(
                    pct_start)
            )

        # Validate anneal_strategy
        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                "anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(
                    anneal_strategy
                )
            )
        elif anneal_strategy == "cos":
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == "linear":
            self.anneal_func = self._annealing_linear

        # Initialize learning rate variables
        self.initial_lr = max_lr / div_factor
        self.max_lr = max_lr
        self.min_lr = self.initial_lr / final_div_factor

        # Initial momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            self.m_momentum = max_momentum
            self.momentum = max_momentum
            self.b_momentum = base_momentum

        # Initialize variable to learning_rate & momentum
        self.track_lr = []
        self.track_mom = []

    def _annealing_cos(self, start, end, pct) -> float:
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct) -> float:
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def set_lr_mom(self) -> None:
        """Update the learning rate and momentum"""
        if self.step_num <= self.step_size_up:
            # update learining rate
            computed_lr = self.anneal_func(
                self.initial_lr, self.max_lr, self.step_num / self.step_size_up
            )
            K.set_value(self.model.optimizer.lr, computed_lr)
            # update momentum if cycle_momentum
            if self.cycle_momentum:
                computed_momentum = self.anneal_func(
                    self.m_momentum, self.b_momentum, self.step_num / self.step_size_up
                )
                try:
                    K.set_value(self.model.optimizer.momentum,
                                computed_momentum)
                except:
                    K.set_value(self.model.optimizer.beta_1, computed_momentum)
        else:
            down_step_num = self.step_num - self.step_size_up
            # update learning rate
            computed_lr = self.anneal_func(
                self.max_lr, self.min_lr, down_step_num / self.step_size_down
            )
            K.set_value(self.model.optimizer.lr, computed_lr)
            # update momentum if cycle_momentum
            if self.cycle_momentum:
                computed_momentum = self.anneal_func(
                    self.b_momentum,
                    self.m_momentum,
                    down_step_num / self.step_size_down,
                )
                try:
                    K.set_value(self.model.optimizer.momentum,
                                computed_momentum)
                except:
                    K.set_value(self.model.optimizer.beta_1, computed_momentum)

    def on_train_begin(self, logs=None) -> None:
        # Set initial learning rate & momentum values
        K.set_value(self.model.optimizer.lr, self.initial_lr)
        if self.cycle_momentum:
            try:
                K.set_value(self.model.optimizer.momentum, self.momentum)
            except:
                K.set_value(self.model.optimizer.beta_1, self.momentum)

    def on_train_batch_end(self, batch, logs=None) -> None:
        # Grab the current learning rate & momentum
        lr = float(K.get_value(self.model.optimizer.lr))
        try:
            mom = float(K.get_value(self.model.optimizer.momentum))
        except:
            mom = float(K.get_value(self.model.optimizer.beta_1))
        # Append to the list
        self.track_lr.append(lr)
        self.track_mom.append(mom)
        # Update learning rate & momentum
        self.set_lr_mom()
        # increment step_num
        self.step_num += 1

    def plot_lrs_moms(self, axes=None) -> None:
        if axes == None:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        else:
            try:
                ax1, ax2 = axes
            except:
                ax1, ax2 = axes[0], axes[1]
        ax1.plot(self.track_lr)
        ax1.set_title("Learning Rate vs Steps")
        ax2.plot(self.track_mom)
        ax2.set_title("Momentum (or beta_1) vs Steps")
