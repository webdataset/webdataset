from torch.utils.data import DataLoader
import pytest

import webdataset as wds
from tests.testconfig import count_samples_tuple, local_data, remote_loc, remote_shards


def test_webloader():
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


def test_webloader2():
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
    ds = wds.WebDataset(remote_loc + remote_shards)
    dl = DataLoader(ds, num_workers=4)
    assert count_samples_tuple(dl, n=100) == 100


def test_webloader_repeat():
    ds = wds.WebDataset(local_data, empty_check=False).decode().to_tuple("cls")
    dl = wds.WebLoader(ds, num_workers=4, batch_size=3).repeat(nepochs=2)
    nsamples = count_samples_tuple(dl)
    assert nsamples == 2 * (47 + 2) // 3, nsamples


def test_webloader_unbatched():
    ds = wds.WebDataset(local_data, empty_check=False).decode().to_tuple("cls")
    dl = wds.WebLoader(ds, num_workers=4, batch_size=3).unbatched()
    nsamples = count_samples_tuple(dl)
    assert nsamples == 47, nsamples

def test_check_empty_throws_ValueError():
    with pytest.raises(ValueError):
        ds = wds.WebDataset(local_data).decode().to_tuple("cls")
        dl = wds.WebLoader(ds, num_workers=4, batch_size=3).repeat(nepochs=2)
        nsamples = count_samples_tuple(dl)
    
