from fastai.torch_core import *
from fastai.basic_train import LearnerCallback
from fastai.vision import ConvLearner
from fastai.data import DataBunch
from fastai.metrics import accuracy

from torchvision import transforms
from torchvision.models import resnet34
from torchvision.datasets import CIFAR10


@dataclass
class CSVLogger(LearnerCallback):

    filename:str='history.csv'

    def __post_init__(self):
        self.path = Path(self.filename)
        self.file = None

    @property
    def header(self):
        return self.learn.recorder.names

    def on_train_begin(self, metrics_names:StrList, **kwargs:Any)->None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w')
        self.file.write(','.join(self.header) + '\n')

    def on_epoch_end(self, epoch:int, smooth_loss:Tensor, last_metrics:MetricsList, **kwargs:Any)->bool:
        self.write_stats([epoch, smooth_loss] + last_metrics)

    def on_train_end(self, **kwargs:Any)->None:
        self.file.flush()
        self.file.close()

    def write_stats(self, stats:TensorOrNumList)->None:
        stats = [str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                 for name,stat in zip(self.header,stats)]
        str_stats = ','.join(stats)
        self.file.write(str_stats + '\n')


class Classifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        return F.relu(self.out(x.view(x.size(0), -1)))


def main():
    root = Path.home()/'data'/'CIFAR10'
    tfms = transforms.Compose([transforms.ToTensor()])
    train_ds = CIFAR10(root, train=True, download=True, transform=tfms)
    valid_ds = CIFAR10(root, train=False, download=True, transform=tfms)
    data = DataBunch.create(train_ds, valid_ds)

    setattr(train_ds, 'loss_func', F.cross_entropy)

    learner = ConvLearner(data, resnet34, custom_head=Classifier(), metrics=[accuracy])
    learner.fit(3, callbacks=[CSVLogger(learner)])


if __name__ == '__main__':
    main()
