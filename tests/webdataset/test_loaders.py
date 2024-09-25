import pytest
from torch.utils.data import DataLoader

import webdataset as wds
from tests.conftest import count_samples_tuple, local_data, remote_loc, remote_shards


def test_webloader():
    wds.pytorch_weights_only = True
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.decode(),
        wds.to_tuple("cls"),
    )
    dl = DataLoader(ds, num_workers=4, batch_size=3)
    nsamples = count_samples_tuple(dl)
    assert nsamples == (47 + 2) // 3, nsamples


@pytest.mark.quick
def test_webloader2():
    wds.pytorch_weights_only = True
    ds = wds.DataPipeline(
        wds.SimpleShardList(local_data),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.decode("torchrgb"),
        wds.to_tuple("cls"),
    )
    dl = wds.DataPipeline(
        DataLoader(ds, num_workers=4, batch_size=3, drop_last=True),
        wds.unbatched(),
    )
    nsamples = count_samples_tuple(dl)
    assert nsamples == 45, nsamples


def test_dataloader():
    wds.pytorch_weights_only = True
    ds = wds.WebDataset(remote_loc + remote_shards, shardshuffle=False)
    dl = DataLoader(ds, num_workers=4)
    assert count_samples_tuple(dl, n=100) == 100


def test_webloader_repeat():
    wds.pytorch_weights_only = True
    ds = (
        wds.WebDataset(local_data, empty_check=False, shardshuffle=False)
        .decode()
        .to_tuple("cls")
    )
    dl = wds.WebLoader(ds, num_workers=4, batch_size=3).repeat(nepochs=2)
    nsamples = count_samples_tuple(dl)
    assert nsamples == 2 * (47 + 2) // 3, nsamples


def test_webloader_unbatched():
    wds.pytorch_weights_only = True
    ds = (
        wds.WebDataset(local_data, empty_check=False, shardshuffle=False)
        .decode()
        .to_tuple("cls")
    )
    dl = wds.WebLoader(ds, num_workers=4, batch_size=3).unbatched()
    nsamples = count_samples_tuple(dl)
    assert nsamples == 47, nsamples


def test_check_empty_throws_ValueError():
    wds.pytorch_weights_only = True
    with pytest.raises(ValueError):
        ds = wds.WebDataset(local_data, shardshuffle=False).decode().to_tuple("cls")
        dl = wds.WebLoader(ds, num_workers=4, batch_size=3).repeat(nepochs=2)
        nsamples = count_samples_tuple(dl)
