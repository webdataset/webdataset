# webdataset.tenbin

Binary tensor encodings for PyTorch and NumPy.

This defines efficient binary encodings for tensors. The format is 8 byte
aligned and can be used directly for computations when transmitted, say,
via RDMA. The format is supported by WebDataset with the `.ten` filename
extension. It is also used by Tensorcom, Tensorcom RDMA, and can be used
for fast tensor storage with LMDB and in disk files (which can be memory
mapped)

Data is encoded as a series of chunks:

:param magic number (int64)
:param length in bytes (int64)
:param bytes (multiple of 64 bytes long)

Arrays are a header chunk followed by a data chunk.
Header chunks have the following structure:

:param dtype (int64)
:param 8 byte array name
:param ndim (int64)
:param dim[0]
:param dim[1]
:param ...

## write
```python
write(stream, l, infos=None)
```
Write a list of arrays to a stream, with magics, length, and padding.
## read
```python
read(stream, n=999999, infos=False)
```
Read a list of arrays from a stream, with magics, length, and padding.
## save
```python
save(fname, *args, infos=None, nocheck=False)
```
Save a list of arrays to a file, with magics, length, and padding.
## load
```python
load(fname, infos=False, nocheck=False)
```
Read a list of arrays from a file, with magics, length, and padding.
## zsend_single
```python
zsend_single(socket, l, infos=None)
```
Send arrays as a single part ZMQ message.
## zrecv_single
```python
zrecv_single(socket, infos=False)
```
Receive arrays as a single part ZMQ message.
## zsend_multipart
```python
zsend_multipart(socket, l, infos=None)
```
Send arrays as a multipart ZMQ message.
## zrecv_multipart
```python
zrecv_multipart(socket, infos=False)
```
Receive arrays as a multipart ZMQ message.
## sctp_send
```python
sctp_send(socket, dest, l, infos=None)
```
Send arrays as an SCTP datagram.

This is just a convenience function and illustration.
For more complex networking needs, you may want
to call encode_buffer and sctp_send directly.

## sctp_recv
```python
sctp_recv(socket, infos=False, maxsize=100000000)
```
Receive arrays as an SCTP datagram.

This is just a convenience function and illustration.
For more complex networking needs, you may want
to call sctp_recv and decode_buffer directly.

