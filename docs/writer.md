# webdataset.writer

## TarWriter
```python
TarWriter(self, fileobj, keep_meta=False, user='bigdata', group='bigdata', mode=292, compress=None, encoder=True)
```
A class for writing dictionaries to tar files.

- fileobj: fileobj: file name for tar file (.tgz/.tar) or open file descriptor
- bool: keep_meta: keep fields starting with "_"
- keep_meta:  (Default value = False)
- encoder: sample encoding (Default value = None)
- compress:  (Default value = None)

The following code will add two file to the tar archive: `a/b.png` and
`a/b.output.png`.

```Python
    tarwriter = TarWriter(stream)
    image = imread("b.jpg")
    image2 = imread("b.out.jpg")
    sample = {"__key__": "a/b", "png": image, "output.png": image2}
    tarwriter.write(sample)
```

## ShardWriter
```python
ShardWriter(self, pattern, maxcount=100000, maxsize=3000000000.0, keep_meta=False, user=None, group=None, compress=None, post=None, **kw)
```
Like TarWriter but splits into multiple shards.

- pattern: output file pattern
- maxcount: maximum number of records per shard (Default value = 100000)
- maxsize: maximum size of each shard (Default value = 3e9)
- kw: other options passed to TarWriter


