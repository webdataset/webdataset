"""Streaming tar metadata must not accumulate when entries are skipped."""

import io
import tarfile

import pytest

from webdataset.tariterators import tar_file_iterator


@pytest.mark.quick
@pytest.mark.parametrize("mode", ["w", "w:gz"])
@pytest.mark.parametrize("skip", ["select", "metadata", "directory", "rename_error", "select_error", "none"])
def test_filtered_tar_member_retention(monkeypatch, mode, skip):
    archive = io.BytesIO()
    expected = []
    with tarfile.open(fileobj=archive, mode=mode) as output:
        for i in range(32):
            name = f"skip-{i}.txt"
            if skip == "metadata":
                name = f"__metadata__/{i}.txt"
            member = tarfile.TarInfo(name)
            if skip == "directory":
                member.type = tarfile.DIRTYPE
                output.addfile(member)
            else:
                member.size = 1
                output.addfile(member, io.BytesIO(b"x"))
            if skip == "none":
                expected.append({"fname": name, "data": b"x"})
        member = tarfile.TarInfo("keep.txt")
        member.size = 4
        output.addfile(member, io.BytesIO(b"keep"))
        expected.append({"fname": "keep.txt", "data": b"keep"})
    archive.seek(0)

    retained = []

    class TrackingTarFile(tarfile.TarFile):
        def __iter__(self):
            for member in super().__iter__():
                retained.append(len(self.members))
                yield member

    monkeypatch.setattr(tarfile, "open", TrackingTarFile.open)
    errors = []

    def handle(error):
        errors.append(error)
        return True

    def reject(name):
        if name != "keep.txt":
            raise ValueError("skip synthetic entry")
        return name

    options = {"handler": handle}
    if skip == "select":
        options["select_files"] = lambda name: name == "keep.txt"
    elif skip == "rename_error":
        options["rename_files"] = reject
    elif skip == "select_error":
        options["select_files"] = reject

    assert list(tar_file_iterator(archive, **options)) == expected
    assert len(errors) == (32 if skip.endswith("error") else 0)
    assert len(retained) == 33
    assert max(retained) <= 1
