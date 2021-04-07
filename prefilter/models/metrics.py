import pytorch_lightning as pl

__all__ = ['configure_metrics']

def configure_metrics():

    metric_collection = pl.metrics.MetricCollection([
        pl.metrics.Accuracy(),
        ])

    return metric_collection
