import tensorflow as tf
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from loguru import logger


class ReduceLRWithRestore(tf.keras.callbacks.Callback):
    def __init__(self, lr_monitor="val_loss", restore_monitor="val_loss",
                 patience=5, factor=0.2, min_lr=1e-6, min_delta=1e-4, cooldown=1, verbose=1):
        super().__init__()
        self.lr_monitor = lr_monitor
        self.restore_monitor = restore_monitor
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.verbose = verbose

        self.best_lr_val = float("inf")
        self.wait = 0
        self.cooldown_counter = 0
        self.reductions = 0

        self.best_restore_val = float("inf")
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr_current = logs.get(self.lr_monitor)
        restore_current = logs.get(self.restore_monitor)

        if lr_current is None or restore_current is None:
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        if restore_current < self.best_restore_val:
            self.best_restore_val = restore_current
            self.best_weights = self.model.get_weights()

        if lr_current < (self.best_lr_val - self.min_delta):
            self.best_lr_val = lr_current
            self.wait = 0
            return

        if self.cooldown_counter > 0:
            return

        self.wait += 1
        if self.wait >= self.patience:
            old_lr = float(self.model.optimizer.learning_rate)
            if old_lr <= self.min_lr + 1e-12:
                self.wait = 0
                return

            new_lr = max(old_lr * self.factor, self.min_lr)

            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)

            self.model.optimizer.learning_rate.assign(new_lr)
            self.reductions += 1
            self.cooldown_counter = self.cooldown
            self.wait = 0

            if self.verbose:
                logger.info(
                    f"⚡ Epoch {epoch+1}: rollback(best {self.restore_monitor}={self.best_restore_val:.4f}) "
                    f"+ LR {old_lr:.2e} → {new_lr:.2e} (#{self.reductions})"
                )

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            if self.verbose:
                logger.info(
                    f"🏁 Entrenamiento completado. Restaurados pesos con best {self.restore_monitor}={self.best_restore_val:.4f}"
                )


class KeepBestValBalancedAcc(tf.keras.callbacks.Callback):
    def __init__(self, val_inputs, val_targets, restore_on_train_end=True, verbose=1):
        super().__init__()
        self.val_inputs = val_inputs
        self.val_targets = np.asarray(val_targets).astype(int)
        self.restore_on_train_end = restore_on_train_end
        self.verbose = verbose

        self.best_score = -float("inf")
        self.best_epoch = -1
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        probs = self.model.predict(self.val_inputs, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        score = float(balanced_accuracy_score(self.val_targets, y_pred))
        logs["val_balanced_acc"] = score

        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            if self.verbose:
                logger.info(f"🏅 Epoch {epoch+1}: nuevo best val_balanced_acc={score:.4f}")

    def on_train_end(self, logs=None):
        if self.restore_on_train_end and self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            if self.verbose:
                logger.info(
                    f"🎯 Restaurados pesos con best val_balanced_acc={self.best_score:.4f} (epoch {self.best_epoch+1})"
                )
