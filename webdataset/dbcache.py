import sys
import torch
from torch.utils.data import IterableDataset
import sqlite3
import io
import uuid


def get_uuid(data):
    return str(uuid.uuid3(uuid.NAMESPACE_URL, data))


class DBCache(IterableDataset):
    def __init__(self, dbname, size, source=None, shuffle=False, verbose=True):
        self.dbname = dbname
        self.source = source
        self.verbose = verbose
        if dbname is None:
            return
        self.db = sqlite3.connect(dbname)
        self.shuffle = shuffle
        self.db.execute(
            """create table if not exists cache (key text primary key, data blob)"""
        )
        self.db.execute("""create table if not exists meta (key text, value text)""")
        self.total = self.db.execute("select count(*) from cache").fetchone()[0]
        if self.getmeta("size") is not None:
            self.size = int(self.getmeta("size"))
        else:
            self.size = size
        if self.verbose:
            print(f"[DBCache opened {dbname} size {self.size} total {self.total}]", file=sys.stderr, flush=True)

    def __call__(self, source):
        self.source = source
        return self

    def getmeta(self, key):
        l = list(self.db.execute("select value from meta where key=?", (key,)))
        if len(l) == 0:
            return None
        assert len(l) == 1
        return l[0][0]

    def setmeta(self, key, value):
        self.db.execute(
            "insert or replace into meta (key, value) values (?, ?)",
            (str(key), str(value)),
        )

    def __len__(self):
        return self.size

    def dbiter(self):
        if self.verbose:
            print(f"[DBCache starting dbiter total {self.total} size {self.size}]", file=sys.stderr, flush=True)
        query = "select data from cache"
        if self.shuffle:
            query += " order by random()"
        with self.db:
            for (data,) in self.db.execute(query):
                obj = torch.load(io.BytesIO(data))
                yield obj

    def key_exists(self, key):
        return self.db.execute(
            "select exists(select key from cache where key = ? limit 1)", (key,)
        ).fetchone()[0]

    def __iter__(self):

        if self.dbname is None:
            yield from iter(self.source)
            return

        if self.total >= self.size:
            yield from self.dbiter()
            return

        if self.verbose:
            print(f"[DBCache total {self.total} size {self.size} more caching]", file=sys.stderr, flush=True)

        for sample in self.source:
            if self.total >= self.size:
                break
            stream = io.BytesIO()
            torch.save(sample, stream)
            data = stream.getbuffer()
            key = sample.get("__key__") if "__key__" in sample else get_uuid(data)
            if not self.key_exists(key):
                self.db.execute(
                    "insert into cache (key, data) values (?, ?)", (key, data)
                )
                self.total += 1
                if self.total % 10 == 0:
                    self.db.commit()
            yield sample

        self.db.commit()

        if self.verbose:
            print(f"[DBCache finished caching total {self.total} (size {self.size})]", file=sys.stderr, flush=True)
            self.setmeta("size", self.total)
