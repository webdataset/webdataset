```python
%pylab inline

import torch
import webdataset as wds
import braceexpand
```

    Populating the interactive namespace from numpy and matplotlib


# Splitting Shards across Nodes and Workers

Unlike traditional PyTorch `Dataset` instances, `WebDataset` splits data across nodes at the shard level, not at the sample level.

This functionality is handled inside the `ShardList` class. Recall that `dataset = webdataset.Webdataset(urls)` is just a shorthand for:


```python
urls = list(braceexpand.braceexpand("dataset-{000000..000999}.tar"))
dataset = wds.ShardList(urls, splitter=wds.split_by_worker, nodesplitter=wds.split_by_node, shuffle=False)
dataset = wds.Processor(dataset, wds.url_opener)
dataset = wds.Processor(dataset, wds.tar_file_expander)
dataset = wds.Processor(dataset, wds.group_by_keys)
```

Here, `nodesplitter` and `splitter` are functions that are called inside `ShardList` to split up the URLs in `urls` by node and worker. You can use any functions you like there, all they need to do is take a list of URLs and return a subset of those URLs as a result.

The default `split_by_worker` looks roughly like:


```python
def my_split_by_worker(urls):
    wi = torch.utils.data.get_worker_info()
    if wi is None:
        return urls
    else:
        return urls[wi.id::wi.num_workers]
```

The same approach works for multiple worker nodes:


```python
def my_split_by_node(urls):
    node_id, node_count = torch.distributed.get_rank(), torch.distributed.get_world_size()
    return urls[node_id::node_count]
```


```python
dataset = wds.WebDataset(urls, splitter=my_split_by_worker, nodesplitter=my_split_by_node)
```

Of course, you can also create more complex splitting strategies if necessary.

# DistributedDataParallel

DistributedDataParallel training requires that each participating node receive exactly the same number of training batches as all others. The `ddp_equalize` method ensures this:


```python
urls = "./shards/imagenet-train-{000000..001281}.tar"
dataset_size, batch_size = 1282000, 64
dataset = wds.WebDataset(urls).decode("pil").shuffle(5000).batched(batch_size, partial=False)
loader = wds.WebLoader(dataset, num_workers=4)
loader = loader.ddp_equalize(dataset_size // batch_size)
```

You need to give the total number of batches in your dataset to `ddp_equalize`; it will compute the batches per node from this and equalize batches accordingly.

You need to apply `ddp_equalize` to the `WebLoader` rather than the `Dataset`.


```python

```
