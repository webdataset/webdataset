def pytest_configure(config):
    config.addinivalue_line("markers", "quick: mark test as quick to run")
    config.addinivalue_line("markers", "notorch: mark test as not requiring torch")


def count_samples_tuple(source, *args, n=10000):
    count = 0
    for i, sample in enumerate(iter(source)):
        if i >= n:
            break
        assert isinstance(sample, (tuple, dict, list)), (type(sample), sample)
        for f in args:
            assert f(sample)
        count += 1
    return count


# FIXME: convert these to test fixtures

local_data = "testdata/imagenet-000000.tgz"
compressed = "testdata/compressed.tar"
remote_loc = "http://storage.googleapis.com/webdataset/openimages/"
remote_shards = "openimages-train-0000{00..99}.tar"
remote_shard = "openimages-train-000321.tar"
remote_pattern = "openimages-train-{}.tar"
remote_sample = "http://storage.googleapis.com/webdataset/testdata/sample.tgz"
