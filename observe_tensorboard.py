from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("lightning_logs", name="exp1")
trainer = pl.Trainer(logger=logger, max_epochs=100)

# tensorboard --logdir lightning_logs