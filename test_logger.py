from unittest import mock

import pytest
from fastai import *
from fastai.vision import *
from fastai.metrics import *
from torchvision.models import resnet18

from logger import CSVLogger


def test_callback_has_required_properties_after_init(history):
    cb = CSVLogger(mock.Mock(), filename=history)

    assert cb.filename
    assert not cb.path.exists()
    assert cb.file is None


def test_callback_writes_learn_metrics_during_training(classifier_and_logger):
    n_epochs = 3
    classifier, cb = classifier_and_logger

    classifier.fit(n_epochs, callbacks=[cb])

    log_df = cb.read_logged_file()
    assert cb.path.exists()
    assert cb.file.closed
    assert not log_df.empty
    assert len(log_df) == n_epochs
    assert classifier.recorder.names == log_df.columns.tolist()


def test_callback_works_with_fit_one_cycle_method(classifier_and_logger, monkeypatch):

    class MockFit:
        def __init__(self):
            self.n_callbacks = 0

        def __call__(self, *args, **kwargs):
            self.n_callbacks = len(kwargs['callbacks'])


    classifier, cb = classifier_and_logger
    mock_fit = MockFit()
    setattr(classifier, 'fit', mock_fit)

    classifier.fit_one_cycle(1, callbacks=[cb])

    assert mock_fit.n_callbacks == 2


@pytest.fixture
def history(tmpdir): return tmpdir.join('history.csv')


@pytest.fixture
def classifier(history):
    path = untar_data(URLs.MNIST_TINY)
    bunch = ImageDataBunch.from_folder(path)
    learn = ConvLearner(bunch, resnet18)
    return learn


@pytest.fixture
def classifier_and_logger(classifier, history):
    classifier.metrics = [accuracy, error_rate]
    cb = CSVLogger(classifier, filename=history)
    return classifier, cb


@pytest.fixture
def patched_fit(monkeypatch):
    monkeypatch.setattr()