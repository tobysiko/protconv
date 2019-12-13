from keras.callbacks import   Callback

class TimeHistory(Callback):
    def __init__(self):
        super(TimeHistory, self).__init__()

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        time_diff = time.time() - self.epoch_time_start
        logs["epoch_duration"] = time_diff
        self.times.append(time_diff)


class ModelCheckpointMultiGPU(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        cpu_model: the original model before it is distributed to multiple GPUs
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.    
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(
        self,
        filepath,
        cpu_model,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        period=1,
    ):
        super(ModelCheckpointMultiGPU, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        self.cpu_model = cpu_model

        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                "ModelCheckpoint mode %s is unknown, "
                "fallback to auto mode." % (mode),
                RuntimeWarning,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        "Can save best model only with %s available, "
                        "skipping." % (self.monitor),
                        RuntimeWarning,
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                "Epoch %05d: %s improved from %0.5f to %0.5f,"
                                " saving model to %s"
                                % (
                                    epoch + 1,
                                    self.monitor,
                                    self.best,
                                    current,
                                    filepath,
                                )
                            )
                        self.best = current
                        if self.save_weights_only:
                            self.cpu_model.save_weights(filepath, overwrite=True)
                        else:
                            self.cpu_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print(
                                "Epoch %05d: %s did not improve"
                                % (epoch + 1, self.monitor)
                            )
            else:
                if self.verbose > 0:
                    print("Epoch %05d: saving model to %s" % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.cpu_model.save_weights(filepath, overwrite=True)
                else:
                    self.cpu_model.save(filepath, overwrite=True)

def plot_keras_model_graph(my_model, outdir, modlabel="my_model",rankdir="TB"):
    
    plot_model(
        my_model,
        to_file=os.path.join(outdir, modlabel + "_model-diagram.png"),
        show_shapes=True,
        show_layer_names=True,
        rankdir=rankdir,
    )  #'LR'