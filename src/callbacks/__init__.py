import pytorch_lightning as pl


class CallbackSet:

    _callbacks = []

    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best_loss_model",
        auto_insert_metric_name=False,
        save_top_k=1,
    )
    _callbacks.append(checkpoint_callback)

    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="epoch_{epoch}_{val_loss:.5f}",
        auto_insert_metric_name=False,
        save_top_k=10,
    )

    _callbacks.append(checkpoint_callback)

    def __init__(self):
        pass

    @staticmethod
    def callbacks():
        return CallbackSet._callbacks
