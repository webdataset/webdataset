import webdataset as wds
import multiprocessing as mp
import zmq
import pickle
import weakref
import uuid


the_protocol = pickle.HIGHEST_PROTOCOL

all_pids = weakref.WeakSet()


class Finished:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def reader(dsfun, urls, sockname, keywords, index=-1):
    global the_protocol
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(sockname)
    ds = dsfun(urls, **keywords)
    for sample in ds:
        data = pickle.dumps(sample, protocol=the_protocol)
        sock.send(data)
    sock.send(pickle.dumps(Finished(index=index)))
    sock.close()


class MultiLoader:

    def __init__(self, dataset, workers=4, verbose=False, keepold=False):
        self.dataset = dataset
        self.keywords = keywords
        self.workers = workers
        self.verbose = verbose
        self.pids = []
        self.socket = None
        self.ctx = zmq.Context.instance()
        self.keepold = keepold

    def kill(self):
        for pid in self.pids:
            if pid is None:
                continue
            print("killing", pid)
            pid.kill()
            pid.join(1.0)
        self.pids = []
        if self.socket is not None:
            print("closing", self.socket)
            self.socket.close()
        self.socket = None

    def __iter__(self):
        if not self.keepold:
            self.kill()
        self.sockname = "ipc://" + str(uuid.uuid4())
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.bind(self.sockname)
        if self.verbose:
            print("#", self.sockname)
        self.pids = [None] * self.workers
        for index in range(self.workers):
            args = (self.dsfun, self.urls, self.sockname, self.keywords, index)
            self.pids[index] = mp.Process(target=reader, args=args)
        all_pids.update(self.pids)
        for pid in self.pids:
            pid.start()
        count = 0
        while self.pids.count(None) < len(self.pids):
            data = self.socket.recv()
            sample = pickle.loads(data)
            if isinstance(sample, Finished):
                if self.verbose:
                    print("# subprocess finished", sample.index)
                self.pids[sample.index].join(1.0)
                self.pids[sample.index] = None
            else:
                yield sample
            count += 1


class DistSender:
    def __init__(self, sockname):
        self.sockname = sockname
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.connect(sockname)

    def send(self, sample):
        data = pickle.dumps(sample, protocol=the_protocol)
        self.sock.send(data)


class DistLoader:
    def __init__(self, sockname):
        self.sockname = sockname

    def __iter__(self):
        self.ctx = zmq.Context.instance()
        sock = self.ctx.socket(zmq.PULL)
        sock.bind(self.sockname)
        while True:
            data = sock.recv()
            sample = pickle.loads(data)
            yield sample
