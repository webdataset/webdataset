import io
import os
import pickle
from io import StringIO
from itertools import islice

import numpy as np
import PIL
import pytest
import yaml

import webdataset as wds
from tests.conftest import (
    compressed,
    count_samples_tuple,
    local_data,
    remote_loc,
    remote_shards,
)
from webdataset import autodecode, handlers, shardlists


@pytest.mark.skip(reason="obsolete")
def test_shardspec():
    dataset = wds.DataPipeline(
        wds.shardspec("testdata/imagenet-000000.tgz"),
        wds.tarfile_samples,
        wds.decode(autodecode.ImageHandler("rgb")),
    )
    result = list(iter(dataset))
    keys = list(result[0].keys())
    assert "__key__" in keys
    assert "__url__" in keys
    assert "cls" in keys
    assert "png" in keys
    assert isinstance(result[0]["cls"], int)
    assert isinstance(result[0]["png"], np.ndarray)
    assert result[0]["png"].shape == (793, 600, 3)
    assert len(result) == 47


shardspec = """
datasets:

  - name: CDIP
    perepoch: 10
    buckets:
      - ./gs/nvdata-ocropus/words/
    shards:
    - cdipsub-{000000..000092}.tar

  - name: Google 1000 Books
    perepoch: 20
    buckets:
      - ./gs/nvdata-ocropus/words/
    shards:
      - gsub-{000000..000167}.tar

  - name: Internet Archive Sample
    perepoch: 30
    buckets:
      - ./gs/nvdata-ocropus/words/
    shards:
      - ia1-{000000..000033}.tar
"""

os.environ["ALLOW_OBSOLETE"] = "1"

yaml3_data = """
prefix: pipe:curl -s -L http://storage.googleapis.com/
buckets: ocropus4-data
datasets:
  - shards: ia1/tess/ia1-{000000..000033}.tar
  - shards: gsub/tess/gsub-{000000..000167}.tar
  - shards: cdipsub/tess/cdipsub-{000000..000092}.tar
"""


@pytest.mark.skip(reason="remote data is inaccessible and yaml spec may be deprecated")
def test_yaml3():
    """Create a WebDataset from a YAML spec.

    The spec is a list of datasets, each of which is a list of shards.
    """
    spec = yaml.safe_load(StringIO(yaml3_data))
    ds = wds.WebDataset(spec, shardshuffle=False)
    next(iter(ds))


shardspec = """
datasets:

  - name: CDIP
    resample: 10
    buckets: ./gs/nvdata-ocropus/words/
    shards: cdipsub-{000000..000092}.tar

  - name: Google 1000 Books
    choose: 20
    buckets:
      - ./gs/nvdata-ocropus/words/
    shards:
      - gsub-{000000..000167}.tar

  - name: Internet Archive Sample
    resample: 30
    buckets:
      - ./gs/nvdata-ocropus/words/
    shards:
      - ia1-{000000..000033}.tar
"""


@pytest.mark.skip(reason="obsolete")
def test_yaml(tmp_path):
    tmp_path = str(tmp_path)
    fname = tmp_path + "/test.shards.yml"
    with open(fname, "w") as stream:
        stream.write(shardspec)
    ds = wds.MultiShardSample(fname)
    l = list(iter(ds))
    assert len(l) == 60, len(l)


@pytest.mark.skip(reason="obsolete")
def test_yaml2():
    spec = yaml.safe_load(StringIO(shardspec))
    ds = wds.MultiShardSample(spec)
    l = list(iter(ds))
    assert len(l) == 60, len(l)
