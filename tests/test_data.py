from collections import Counter

from mnist.data.dataloader import make_training_dataloader


def test_data():
    batch_size = 100
    dataloader = make_training_dataloader(batch_size)
    assert len(dataloader.dataset) == 50000
    label_counter = Counter()

    for data, label in dataloader:
        assert data.shape == (batch_size, 1, 28, 28)
        label_counter.update(label.tolist())

    assert len(label_counter) == 10  # we have 10 diffrent digits in the dataset
