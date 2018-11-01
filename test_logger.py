from io import StringIO
from contextlib import redirect_stdout

import pytest
from fastai import *
from fastai.vision import *
from fastai.metrics import *
from fastprogress import fastprogress

from logger import CSVLogger


def test_callback_has_required_properties_after_init(classifier):
    cb = CSVLogger(classifier)

    assert cb.filename
    assert not cb.path.exists()
    assert cb.learn is classifier
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


def test_callback_written_metrics_are_equal_to_reported_via_stdout(classifier_and_logger, no_bar):
    n_epochs = 3
    classifier, cb = classifier_and_logger

    buffer = StringIO()
    with redirect_stdout(buffer):
        classifier.fit(n_epochs, callbacks=[cb])

    csv_df = cb.read_logged_file()
    stdout_df = convert_into_dataframe(buffer)
    pd.testing.assert_frame_equal(csv_df, stdout_df)


def test_callback_written_metrics_are_equal_to_values_stored_in_reporter(classifier_and_logger):
    n_epochs = 3
    classifier, cb = classifier_and_logger

    classifier.fit(n_epochs, callbacks=[cb])

    csv_df = cb.read_logged_file()
    recorder_df = create_metrics_dataframe(classifier)
    pd.testing.assert_frame_equal(csv_df, recorder_df)


@pytest.fixture
def classifier(tmpdir):
    path = untar_data(URLs.MNIST_TINY)
    bunch = ImageDataBunch.from_folder(path)
    model_path = str(tmpdir.join('classifier'))
    learn = Learner(bunch, simple_cnn((3, 10, 10)), path=model_path)
    return learn


@pytest.fixture
def classifier_and_logger(classifier):
    classifier.metrics = [accuracy, error_rate]
    cb = CSVLogger(classifier)
    return classifier, cb


@pytest.fixture
def no_bar():
    fastprogress.NO_BAR = True
    yield
    fastprogress.NO_BAR = False


def convert_into_dataframe(buffer):
    "Converts data captured from `fastprogress.ConsoleProgressBar` into dataframe."
    lines = buffer.getvalue().split('\n')
    header, *lines = [l.strip() for l in lines if l]
    header = header.split()
    floats = [[float(x) for x in line.split()] for line in lines]
    records = [dict(zip(header, metrics_list)) for metrics_list in floats]
    df = pd.DataFrame(records, columns=header)
    df['epoch'] = df['epoch'].astype(int)
    return df


def create_metrics_dataframe(learn):
    "Converts metrics stored in `Recorder` into dataframe."
    records = [
        [i, loss, val_loss, *epoch_metrics]
        for i, (loss, val_loss, epoch_metrics)
        in enumerate(zip(
            get_train_losses(learn),
            learn.recorder.val_losses,
            learn.recorder.metrics), 1)]
    return pd.DataFrame(records, columns=learn.recorder.names)


def get_train_losses(learn):
    "Returns list of training losses at the end of each training epoch."
    np_losses = [to_np(l).item() for l in learn.recorder.losses]
    batch_size = len(learn.data.train_dl)
    return [batch[-1] for batch in partition(np_losses, batch_size)]
