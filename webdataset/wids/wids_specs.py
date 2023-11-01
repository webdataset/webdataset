import json
import io

def load_remote_shardlist(source):
    """Load a remote or local dataset description in JSON format,
    using the Python web client APIs."""

    if isinstance(source, str):
        with open(source) as stream:
            dsdesc = json.load(stream)
    elif isinstance(source, io.IOBase):
        dsdesc = json.load(source)
    else:
        # FIXME: use gopen
        import requests

        jsondata = requests.get(url).text
        dsdesc = json.loads(jsondata)

    return extract_shardlist(dsdesc)


def check_shards(l):
    """Check that a list of shards is well-formed.

    This checks that the list is a list of dictionaries, and that
    each dictionary has a "url" and a "nsamples" key.
    """
    assert isinstance(l, list)
    for shard in l:
        assert isinstance(shard, dict)
        assert "url" in shard
        assert "nsamples" in shard
    return l


def set_all(l, k, v):
    """Set a key to a value in a list of dictionaries."""
    if v is None:
        return
    for x in l:
        if k not in x:
            x[k] = v


def extract_shardlist(dsdesc):
    """Extract a list of shards from a dataset description.
    Dataset descriptions are JSON files. They must have the following format;

    {
        "wids_version": 1,
        # optional immediate shardlist
        "shardlist": [
            {"url": "http://example.com/file.tar", "nsamples": 1000},
            ...
        ],
        # sub-datasets
        "datasets": [
            {"source_url": "http://example.com/dataset.json"},
            {"shardlist": [
                {"url": "http://example.com/file.tar", "nsamples": 1000},
                ...
            ]}
            ...
        ]
    }
    """
    assert isinstance(dsdesc, dict)
    shardlist = dsdesc.get("shardlist", [])
    set_all(shardlist, "weight", dsdesc.get("weight"))
    check_shards(shardlist)
    assert "wids_version" in dsdesc, "No wids_version in dataset description"
    assert dsdesc["wids_version"] == 1, "Unknown wids_version"
    for component in dsdesc.get("datasets", []):
        for i in range(10):
            if "source_url" not in component:
                break
            component = load_remote(component["source_url"])
        assert i < 9, "Too many levels of indirection"
        if "shardlist" in component:
            l = check_shards(component["shardlist"])
            set_all(l, "weight", component.get("weight"))
            shardlist.extend(l)
    assert len(shardlist) > 0, "No shards found"
    return shardlist
