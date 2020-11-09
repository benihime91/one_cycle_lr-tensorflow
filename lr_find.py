import tempfile

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm

K = keras.backend


class Scheduler:
    def __init__(self, vals, n_iter: int) -> None:
        'Used to "step" from start,end (`vals`) over `n_iter` s on a schedule defined by `func`'
        self.start, self.end = (
            (vals[0], vals[1]) if isinstance(vals, tuple) else (vals, 0)
        )
        self.n_iter = max(1, n_iter)
        self.func = self._aannealing_exp
        self.n = 0

    @staticmethod
    def _aannealing_exp(start: float, end: float, pct: float) -> float:
        "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return start * (end / start) ** pct

    def restart(self) -> None:
        self.n = 0

    def step(self) -> float:
        self.n += 1
        return self.func(self.start, self.end, self.n / self.n_iter)

    @property
    def is_done(self) -> bool:
        "Return `True` if schedule completed."
        return self.n >= self.n_iter


class LrFinder:
    """
    [LrFinder Implemetation taken from Fast.ai]
    (https://github.com/fastai/fastai/tree/master/fastai)

    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.

    Args:
    model (tf.keras.Model): wrapped model
    optimizer (tf.keras.optimizers): wrapped optimizer
    loss_fn (tf.keras.losses): loss function

    Example:
        >>> lr_finder = LrFinder(model, optimizer, loss_fn)
        >>> lr_finder.range_test(trn_ds, end_lr=100, num_iter=100)
        >>> lr_finder.plot_lrs() # to inspect the loss-learning rate graph
    """

    def __init__(self,
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_fn: tf.keras.losses.Loss,
                 ) -> None:

        self.lrs = []
        self.losses = []
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.mw = self.model.get_weights()
        self.init_lr = K.get_value(self.optimizer.lr)
        self.iteration = 0
        self.weightsFile = tempfile.mkstemp()[1]

    @tf.function
    def trn_step(self, xb, yb):
        """performs 1 trainig step"""
        with tf.GradientTape() as tape:
            logits = self.model(xb, training=True)
            main_loss = tf.reduce_mean(self.loss_fn(yb, logits))
            loss = tf.add_n([main_loss] + self.model.losses)
        grads = tape.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def range_test(self,
                   trn_ds: tf.data.Dataset,
                   start_lr: float = 1e-7,
                   end_lr: float = 10,
                   num_iter: int = 100,
                   beta=0.98,
                   ) -> None:
        """
        Explore lr from `start_lr` to `end_lr` over `num_it` s in `model`.

        Args:
        trn_ds (tf.data.Dataset)
        start_lr (float, optional): the starting learning rate for the range test.
                Default:1e-07.
        end_lr (float, optional): the maximum learning rate to test. Default: 10.
        num_iter (int, optional): the number of s over which the test
                occurs. Default: 100.
        beta (float, optional): the loss smoothing factor within the [0, 1]
                interval. The loss is smoothed using exponential smoothing.
                Default: 0.98.
        """
        # save original model weights
        try:
            self.model.save_weights(self.weightsFile)
        except:
            print("Unable to save initial weights, weights of model will change. Re-instantiate model to load previous weights ...")
        # start scheduler
        sched = Scheduler((start_lr, end_lr), num_iter)
        avg_loss, best_loss, = 0.0, 0.0
        # set the startig lr
        K.set_value(self.optimizer.lr, sched.start)

        print(f"Finding best initial lr over {num_iter} steps")
        # initialize tqdm bar
        bar = tqdm(iterable=range(num_iter))

        # iterate over the batches
        for (xb, yb) in trn_ds:
            self.iteration += 1
            loss, grads = self.trn_step(xb, yb)
            # compute smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_loss = avg_loss / (1 - beta ** self.iteration)

            # record best loss
            if self.iteration == 1 or smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # stop if loss is exploding
            if sched.is_done or (
                smoothed_loss > 4 * best_loss or np.isnan(smoothed_loss)
            ):
                break

            # append losses and lrs
            self.losses.append(smoothed_loss)
            self.lrs.append(K.get_value(self.optimizer.lr))

            # update weights
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))

            # update lr
            K.set_value(self.optimizer.lr, sched.step())

            # update tqdm
            bar.update(1)

        # clean-up
        bar.close()
        sched.restart()
        self._print_prompt()

    def _print_prompt(self) -> None:
        "Cleanup model weights disturbed during LRFinder exploration."
        try:
            self.model.load_weights(self.weightsFile)
        except:
            print(
                "Unable to load inital weights. Re-instantiate model to load previous weights ...")
        K.set_value(self.optimizer.lr, self.init_lr)
        print(
            "LR Finder is complete, type {LrFinder}.plot_lrs() to see the graph.")

    @staticmethod
    def _split_list(vals, skip_start: int, skip_end: int) -> list:
        return vals[skip_start:-skip_end] if skip_end > 0 else vals[skip_start:]

    def plot_lrs(self,
                 skip_start: int = 10,
                 skip_end: int = 5,
                 suggestion: bool = False,
                 show_grid: bool = False,
                 ) -> None:
        """
        Plot learning rate and losses, trimmed between `skip_start` and `skip_end`.
        Optionally plot and return min gradient
        """
        lrs = self._split_list(self.lrs, skip_start, skip_end)
        losses = self._split_list(self.losses, skip_start, skip_end)
        _, ax = plt.subplots(1, 1)
        ax.plot(lrs, losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale("log")
        if show_grid:
            plt.grid(True, which="both", ls="-")
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.0e"))
        if suggestion:
            try:
                mg = (np.gradient(np.array(losses))).argmin()
            except:
                print(
                    "Failed to compute the gradients, there might not be enough points."
                )
                return
            print(f"Min numerical gradient: {lrs[mg]:.2E}")
            ax.plot(lrs[mg], losses[mg], markersize=10,
                    marker="o", color="red")
            self.min_grad_lr = lrs[mg]
            ml = np.argmin(losses)
            print(f"Min loss divided by 10: {lrs[ml]/10:.2E}")
