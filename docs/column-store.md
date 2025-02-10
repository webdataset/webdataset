# Using WebDataset as a Column Store

Sometimes it is desirable to break up a dataset not just by rows but also by columns.
This is quite easy in WebDataset, although there is no explicit API for it (one will likely be added).

The idea is to just use the `__url__` field in a sample to load additional columns as necessary.


```python
# We usually abbreviate webdataset as wds
import webdataset as wds
```


```python
batchsize = 32
bucket = "https://storage.googleapis.com/webdataset/fake-imagenet"
training_urls = bucket + "/imagenet-train-{000000..001281}.tar"
```


```python
# Create the datasets with shard and sample shuffling and decoding.
trainset = wds.WebDataset(training_urls, resampled=True, shardshuffle=True)

```

This function computes the URL for an additional column from a base URL. This is
then used by the `add_column` function to add data from that additional URL to the
data already loaded from the base URL.


```python
def find_column_url(url):
    # In this function, given the main URL for a shard, find the corresponding
    # extra column URL.

    # For the demo, we just return the same URL, which means that we simply
    # add the same values to the samples twice.
    return url # .replace("-train", "-train-more")
```


```python
def add_column(src, find_column_url=find_column_url):
    """Given an iterator over a dataset, add an extra column from a separate dataset."""
    last_url = None
    column_src = None
    for sample in src:
        # We use the __url__ field to keep track of which shard we are working on.
        # We then open the corresponding URL for the extra column data if necessary.
        if last_url != sample["__url__"]:
            column_url = find_column_url(sample["__url__"])
            print("*** opening column_url", column_url)
            column_src = iter(wds.WebDataset(column_url, shardshuffle=False))
            last_url = sample["__url__"]
        # Read the next sample from the extra column data.
        extra = next(column_src)
        # Check that the keys match.
        assert extra["__key__"] == sample["__key__"]
        # Update the sample with the extra data.
        for k, v in extra.items():
            if k[0] != "_":
                sample[k] = v
        yield sample

trainset = trainset.compose(add_column)

# NB: any shuffling, decoding, etc. needs to happen after the `add_column` call
```

Let's see all of it in action. Actually, nothing particularly interesting happens here
because we are just loading the same data for the base URL and the additional column.
Really, the only feedback you get from this code is the message about opening the column_url.


```python
for k, v in next(iter(trainset)).items():
    print(k, repr(v)[:60])
```

    *** opening column_url https://storage.googleapis.com/webdataset/fake-imagenet/imagenet-train-001010.tar
    __key__ '001010-000002'
    __url__ 'https://storage.googleapis.com/webdataset/fake-imagenet/ima
    cls b'9'
    jpg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x
    txt b'a high quality color photograph of a frog'


Some comments:

- The code above assumes an exact correspondence between the samples in the different columnn shards; this is really what you ought to aim for. But you can add code to skip data.
- For small amounts of data (like class labels), you probably just want to store the data in a dbm-style database and use `.associate(data)`.
- You could also use `wids` to retrieve additional samples in `add_column`.

If you want to do the same thing in `wids`, the code becomes even simpler:

```Python
class CombinedDataset:
    def __init__(self, ds1, ds2):
        self.ds1 = wids.ShardListDataset(ds1)
        self.ds2 = wids.ShardListDataset(ds2)
        assert len(self.ds1) == len(self.ds2)
    def getitem(self, index):
        return self.ds1[index].update(self.ds2[index])
    def __len__(self):
        return len(self.ds1)
```


```python

```
