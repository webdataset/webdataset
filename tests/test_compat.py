from webdataset import compat


def test_shuffle_seed():
    """Test that shuffle is deterministic for a given seed."""

    def make_shuffle_only_ds(seed=0):
        ds = compat.WebDataset(
            "shard-{000000..000999}.tar", shardshuffle=True, seed=seed
        )
        index = ["shuffle" in str(stage) for stage in ds.pipeline].index(True)
        del ds.pipeline[index + 1 :]
        return ds

    ds1 = make_shuffle_only_ds(seed=0)
    ds2 = make_shuffle_only_ds(seed=0)
    assert list(ds1) == list(ds2)
