from fastai import *
from fastai.torch_core import *
from fastai.vision import *
from fastai.metrics import *
from torchvision.models import resnet18


@dataclass
class CSVLogger(LearnerCallback):
    "A `LearnerCallback` that saves history of training metrics into CSV file."
    filename: str = 'history'

    def __post_init__(self):
        self.path = self.learn.path/f'{self.filename}.csv'
        self.file = None

    @property
    def header(self):
        return self.learn.recorder.names

    def read_logged_file(self):
        return pd.read_csv(self.path)

    def on_train_begin(self, metrics_names: StrList, **kwargs: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w')
        self.file.write(','.join(self.header) + '\n')

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        self.write_stats([epoch, smooth_loss] + last_metrics)

    def on_train_end(self, **kwargs: Any) -> None:
        self.file.flush()
        self.file.close()

    def write_stats(self, stats: TensorOrNumList) -> None:
        stats = [str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                 for name, stat in zip(self.header, stats)]
        str_stats = ','.join(stats)
        self.file.write(str_stats + '\n')


def main():
    path = untar_data(URLs.MNIST_TINY)
    data = ImageDataBunch.from_folder(path)
    learn = create_cnn(data, resnet18, metrics=[accuracy, error_rate])
    cb = CSVLogger(learn)
    learn.fit(3, callbacks=[cb])
    print(cb.read_logged_file())



if __name__ == '__main__':
    main()
