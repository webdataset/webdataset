import webdataset as wds
from webdataset import multi


local_data = "testdata/imagenet-000000.tgz"
remote_loc = "http://storage.googleapis.com/nvdata-openimages/"
remote_shards = "openimages-train-0000{00..99}.tar"
remote_shard = "openimages-train-000321.tar"
remote_pattern = "openimages-train-{}.tar"


def identity(x):
    return x


def count_samples_tuple(source, *args, n=1000):
    count = 0
    for i, sample in enumerate(iter(source)):
        if i >= n:
            break
        assert isinstance(sample, (tuple, dict, list)), (type(sample), sample)
        for f in args:
            assert f(sample)
        count += 1
    return count


def count_samples(source, *args, n=1000):
    count = 0
    for i, sample in enumerate(iter(source)):
        if i >= n:
            break
        for f in args:
            assert f(sample)
        count += 1
    return count


def test_multi():
    for k in [1, 4, 17]:
        urls = [f"pipe:cat {local_data} # {i}" for i in range(k)]
        ds = wds.Dataset(urls).decode().shuffle(5).to_tuple("png;jpg cls")
        mds = multi.MultiDataset(ds, workers=4)
        assert count_samples_tuple(mds) == 47*k
