"""Tests for multi-modal WebDataset loading from separate tar files."""

import os

import numpy as np
import pytest

import webdataset as wds
from webdataset import writer
from webdataset.multimodal import MultiModalWebDataset, PairedShardList, paired_tar_expander


def _create_tar(path, samples):
    """Create a tar file with the given samples.

    Args:
        path: Path to the tar file.
        samples: List of dicts with __key__ and extension keys.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with writer.TarWriter(path) as sink:
        for sample in samples:
            sink.write(sample)


def _create_paired_shards(tmpdir, n_shards=2, n_samples_per_shard=3):
    """Create paired image and embedding shards for testing.

    Returns:
        Tuple of (modalities dict, expected keys list).
    """
    all_keys = []
    for shard_idx in range(n_shards):
        image_samples = []
        embedding_samples = []
        for sample_idx in range(n_samples_per_shard):
            key = f"sample_{shard_idx:04d}_{sample_idx:04d}"
            all_keys.append(key)
            image_samples.append(
                {
                    "__key__": key,
                    "jpg": np.zeros((4, 4, 3), dtype=np.uint8),
                }
            )
            embedding_samples.append(
                {
                    "__key__": key,
                    "npy": np.ones((8,), dtype=np.float32),
                }
            )
        _create_tar(f"{tmpdir}/images/shard_{shard_idx:04d}.tar", image_samples)
        _create_tar(f"{tmpdir}/embeddings/shard_{shard_idx:04d}.tar", embedding_samples)

    modalities = {
        "images": f"{tmpdir}/images/shard_{{0000..{n_shards - 1:04d}}}.tar",
        "embeddings": f"{tmpdir}/embeddings/shard_{{0000..{n_shards - 1:04d}}}.tar",
    }
    return modalities, sorted(all_keys)


@pytest.mark.quick
def test_paired_shard_list_basic(tmp_path):
    """Test PairedShardList yields correct paired URL dicts."""
    modalities = {
        "images": ["/data/images/shard_0000.tar", "/data/images/shard_0001.tar"],
        "embeddings": ["/data/embeddings/shard_0000.tar", "/data/embeddings/shard_0001.tar"],
    }
    psl = PairedShardList(modalities)
    results = list(psl)
    assert len(results) == 2
    for r in results:
        assert "urls" in r
        assert "images" in r["urls"]
        assert "embeddings" in r["urls"]

    # Verify pairing by index
    assert results[0]["urls"]["images"] == "/data/images/shard_0000.tar"
    assert results[0]["urls"]["embeddings"] == "/data/embeddings/shard_0000.tar"
    assert results[1]["urls"]["images"] == "/data/images/shard_0001.tar"
    assert results[1]["urls"]["embeddings"] == "/data/embeddings/shard_0001.tar"


@pytest.mark.quick
def test_paired_shard_list_brace_expansion(tmp_path):
    """Test PairedShardList with brace expansion strings."""
    modalities = {
        "a": "/data/a/shard_{0000..0003}.tar",
        "b": "/data/b/shard_{0000..0003}.tar",
    }
    psl = PairedShardList(modalities)
    assert len(psl) == 4
    results = list(psl)
    assert len(results) == 4


@pytest.mark.quick
def test_paired_shard_list_count_mismatch():
    """Test that PairedShardList raises ValueError on shard count mismatch."""
    modalities = {
        "images": ["/data/images/shard_0000.tar", "/data/images/shard_0001.tar"],
        "embeddings": ["/data/embeddings/shard_0000.tar"],
    }
    with pytest.raises(ValueError, match="same number of shards"):
        PairedShardList(modalities)


@pytest.mark.quick
def test_paired_shard_list_shuffle(tmp_path):
    """Test PairedShardList shuffling preserves pairing."""
    modalities = {
        "images": [f"/data/images/shard_{i:04d}.tar" for i in range(10)],
        "embeddings": [f"/data/embeddings/shard_{i:04d}.tar" for i in range(10)],
    }
    psl = PairedShardList(modalities, seed=42)
    results = list(psl)
    assert len(results) == 10

    # Verify pairing: shard index should match between modalities
    for r in results:
        img_url = r["urls"]["images"]
        emb_url = r["urls"]["embeddings"]
        img_idx = img_url.split("shard_")[1].split(".")[0]
        emb_idx = emb_url.split("shard_")[1].split(".")[0]
        assert img_idx == emb_idx, f"Pairing broken: {img_url} vs {emb_url}"

    # Verify shuffling actually happened (with 10 items, seed 42 should change order)
    unshuffled = PairedShardList(modalities)
    unshuffled_results = list(unshuffled)
    assert results != unshuffled_results, "Shuffling didn't change the order"


@pytest.mark.quick
def test_basic_iteration(tmp_path):
    """Test basic multi-modal iteration merges samples correctly."""
    modalities, expected_keys = _create_paired_shards(tmp_path)

    ds = MultiModalWebDataset(
        modalities=modalities,
        shardshuffle=False,
        nodesplitter=None,
        workersplitter=None,
        empty_check=False,
    )

    collected_keys = []
    for sample in ds:
        assert "__key__" in sample
        assert "jpg" in sample
        assert "npy" in sample
        assert "__url_images__" in sample
        assert "__url_embeddings__" in sample
        collected_keys.append(sample["__key__"])

    assert sorted(collected_keys) == expected_keys


@pytest.mark.quick
def test_missing_key_skip(tmp_path):
    """Test missing_key_policy='skip' drops misaligned samples."""
    # Create image shard with keys a, b, c
    _create_tar(
        f"{tmp_path}/images/shard_0000.tar",
        [
            {"__key__": "a", "jpg": np.zeros((2, 2, 3), dtype=np.uint8)},
            {"__key__": "b", "jpg": np.zeros((2, 2, 3), dtype=np.uint8)},
            {"__key__": "c", "jpg": np.zeros((2, 2, 3), dtype=np.uint8)},
        ],
    )
    # Create embedding shard with keys a, c (missing b)
    _create_tar(
        f"{tmp_path}/embeddings/shard_0000.tar",
        [
            {"__key__": "a", "npy": np.ones((4,), dtype=np.float32)},
            {"__key__": "c", "npy": np.ones((4,), dtype=np.float32)},
        ],
    )

    modalities = {
        "images": [f"{tmp_path}/images/shard_0000.tar"],
        "embeddings": [f"{tmp_path}/embeddings/shard_0000.tar"],
    }
    ds = MultiModalWebDataset(
        modalities=modalities,
        shardshuffle=False,
        nodesplitter=None,
        workersplitter=None,
        empty_check=False,
        missing_key_policy="skip",
    )

    results = list(ds)
    keys = [r["__key__"] for r in results]
    assert keys == ["a", "c"], f"Expected ['a', 'c'], got {keys}"


@pytest.mark.quick
def test_missing_key_error(tmp_path):
    """Test missing_key_policy='error' raises on misaligned samples."""
    _create_tar(
        f"{tmp_path}/images/shard_0000.tar",
        [
            {"__key__": "a", "jpg": np.zeros((2, 2, 3), dtype=np.uint8)},
            {"__key__": "b", "jpg": np.zeros((2, 2, 3), dtype=np.uint8)},
        ],
    )
    _create_tar(
        f"{tmp_path}/embeddings/shard_0000.tar",
        [
            {"__key__": "a", "npy": np.ones((4,), dtype=np.float32)},
            {"__key__": "c", "npy": np.ones((4,), dtype=np.float32)},
        ],
    )

    modalities = {
        "images": [f"{tmp_path}/images/shard_0000.tar"],
        "embeddings": [f"{tmp_path}/embeddings/shard_0000.tar"],
    }
    ds = MultiModalWebDataset(
        modalities=modalities,
        shardshuffle=False,
        nodesplitter=None,
        workersplitter=None,
        empty_check=False,
        missing_key_policy="error",
    )

    with pytest.raises(ValueError, match="key mismatch"):
        list(ds)


@pytest.mark.quick
def test_missing_key_partial(tmp_path):
    """Test missing_key_policy='partial' yields partial samples."""
    _create_tar(
        f"{tmp_path}/images/shard_0000.tar",
        [
            {"__key__": "a", "jpg": np.zeros((2, 2, 3), dtype=np.uint8)},
            {"__key__": "b", "jpg": np.zeros((2, 2, 3), dtype=np.uint8)},
            {"__key__": "c", "jpg": np.zeros((2, 2, 3), dtype=np.uint8)},
        ],
    )
    _create_tar(
        f"{tmp_path}/embeddings/shard_0000.tar",
        [
            {"__key__": "a", "npy": np.ones((4,), dtype=np.float32)},
            {"__key__": "c", "npy": np.ones((4,), dtype=np.float32)},
        ],
    )

    modalities = {
        "images": [f"{tmp_path}/images/shard_0000.tar"],
        "embeddings": [f"{tmp_path}/embeddings/shard_0000.tar"],
    }
    ds = MultiModalWebDataset(
        modalities=modalities,
        shardshuffle=False,
        nodesplitter=None,
        workersplitter=None,
        empty_check=False,
        missing_key_policy="partial",
    )

    results = list(ds)
    keys = [r["__key__"] for r in results]
    assert "a" in keys
    assert "b" in keys
    assert "c" in keys

    # 'a' and 'c' should have both modalities
    a_sample = [r for r in results if r["__key__"] == "a"][0]
    assert "jpg" in a_sample
    assert "npy" in a_sample

    # 'b' should only have images
    b_sample = [r for r in results if r["__key__"] == "b"][0]
    assert "jpg" in b_sample
    assert "npy" not in b_sample


@pytest.mark.quick
def test_shard_shuffling(tmp_path):
    """Test that shard shuffling produces same total samples."""
    modalities, expected_keys = _create_paired_shards(tmp_path, n_shards=3, n_samples_per_shard=2)

    ds = MultiModalWebDataset(
        modalities=modalities,
        shardshuffle=10,
        nodesplitter=None,
        workersplitter=None,
        empty_check=False,
    )

    collected_keys = []
    for sample in ds:
        collected_keys.append(sample["__key__"])

    assert sorted(collected_keys) == expected_keys


@pytest.mark.quick
def test_three_modalities(tmp_path):
    """Test with three modalities."""
    for shard_idx in range(2):
        samples_img = []
        samples_emb = []
        samples_txt = []
        for sample_idx in range(3):
            key = f"s_{shard_idx}_{sample_idx}"
            samples_img.append({"__key__": key, "jpg": np.zeros((2, 2, 3), dtype=np.uint8)})
            samples_emb.append({"__key__": key, "npy": np.ones((4,), dtype=np.float32)})
            samples_txt.append({"__key__": key, "txt": f"caption {key}"})
        _create_tar(f"{tmp_path}/images/shard_{shard_idx:04d}.tar", samples_img)
        _create_tar(f"{tmp_path}/embeddings/shard_{shard_idx:04d}.tar", samples_emb)
        _create_tar(f"{tmp_path}/text/shard_{shard_idx:04d}.tar", samples_txt)

    modalities = {
        "images": [f"{tmp_path}/images/shard_{i:04d}.tar" for i in range(2)],
        "embeddings": [f"{tmp_path}/embeddings/shard_{i:04d}.tar" for i in range(2)],
        "text": [f"{tmp_path}/text/shard_{i:04d}.tar" for i in range(2)],
    }

    ds = MultiModalWebDataset(
        modalities=modalities,
        shardshuffle=False,
        nodesplitter=None,
        workersplitter=None,
        empty_check=False,
    )

    results = list(ds)
    assert len(results) == 6
    for sample in results:
        assert "jpg" in sample
        assert "npy" in sample
        assert "txt" in sample
        assert "__url_images__" in sample
        assert "__url_embeddings__" in sample
        assert "__url_text__" in sample


@pytest.mark.quick
def test_extension_collision(tmp_path):
    """Test that duplicate extension keys across modalities raise ValueError."""
    _create_tar(
        f"{tmp_path}/mod_a/shard_0000.tar",
        [
            {"__key__": "a", "txt": "hello from mod_a"},
        ],
    )
    _create_tar(
        f"{tmp_path}/mod_b/shard_0000.tar",
        [
            {"__key__": "a", "txt": "hello from mod_b"},
        ],
    )

    modalities = {
        "mod_a": [f"{tmp_path}/mod_a/shard_0000.tar"],
        "mod_b": [f"{tmp_path}/mod_b/shard_0000.tar"],
    }

    ds = MultiModalWebDataset(
        modalities=modalities,
        shardshuffle=False,
        nodesplitter=None,
        workersplitter=None,
        empty_check=False,
    )

    with pytest.raises(ValueError, match="collides"):
        list(ds)


@pytest.mark.quick
def test_fluid_interface_decode(tmp_path):
    """Test FluidInterface chaining with decode."""
    modalities, _ = _create_paired_shards(tmp_path, n_shards=1, n_samples_per_shard=2)

    ds = MultiModalWebDataset(
        modalities=modalities,
        shardshuffle=False,
        nodesplitter=None,
        workersplitter=None,
        empty_check=False,
    ).decode("rgb")

    for sample in ds:
        assert isinstance(sample["jpg"], np.ndarray)
        assert isinstance(sample["npy"], np.ndarray)
        break


@pytest.mark.quick
def test_fluid_interface_to_tuple(tmp_path):
    """Test FluidInterface chaining with to_tuple."""
    modalities, _ = _create_paired_shards(tmp_path, n_shards=1, n_samples_per_shard=2)

    ds = (
        MultiModalWebDataset(
            modalities=modalities,
            shardshuffle=False,
            nodesplitter=None,
            workersplitter=None,
            empty_check=False,
        )
        .decode("rgb")
        .to_tuple("jpg", "npy")
    )

    for img, emb in ds:
        assert isinstance(img, np.ndarray)
        assert isinstance(emb, np.ndarray)
        break


@pytest.mark.quick
def test_fluid_interface_batched(tmp_path):
    """Test FluidInterface chaining with batched."""
    modalities, _ = _create_paired_shards(tmp_path, n_shards=1, n_samples_per_shard=4)

    ds = (
        MultiModalWebDataset(
            modalities=modalities,
            shardshuffle=False,
            nodesplitter=None,
            workersplitter=None,
            empty_check=False,
        )
        .decode("rgb")
        .to_tuple("jpg", "npy")
        .batched(2)
    )

    count = 0
    for batch in ds:
        imgs, embs = batch
        assert len(imgs) == 2
        assert len(embs) == 2
        count += 1
    assert count == 2


@pytest.mark.quick
def test_with_epoch(tmp_path):
    """Test with_epoch works."""
    modalities, _ = _create_paired_shards(tmp_path, n_shards=1, n_samples_per_shard=5)

    ds = MultiModalWebDataset(
        modalities=modalities,
        shardshuffle=False,
        nodesplitter=None,
        workersplitter=None,
        empty_check=False,
    ).with_epoch(3)

    count = sum(1 for _ in ds)
    assert count == 3


@pytest.mark.quick
def test_repeat(tmp_path):
    """Test repeat works."""
    modalities, _ = _create_paired_shards(tmp_path, n_shards=1, n_samples_per_shard=2)

    ds = MultiModalWebDataset(
        modalities=modalities,
        shardshuffle=False,
        nodesplitter=None,
        workersplitter=None,
        empty_check=False,
    ).repeat(3)

    count = sum(1 for _ in ds)
    assert count == 6  # 2 samples * 3 repeats


@pytest.mark.quick
def test_empty_modality_urls():
    """Test that empty URL list raises ValueError."""
    with pytest.raises(ValueError, match="no shard URLs"):
        PairedShardList({"images": []})


@pytest.mark.quick
def test_paired_shard_list_len():
    """Test __len__ of PairedShardList."""
    psl = PairedShardList(
        {
            "a": ["/shard_0.tar", "/shard_1.tar", "/shard_2.tar"],
            "b": ["/shard_0.tar", "/shard_1.tar", "/shard_2.tar"],
        }
    )
    assert len(psl) == 3


@pytest.mark.quick
def test_exports():
    """Test that multimodal classes are exported from webdataset package."""
    assert hasattr(wds, "MultiModalWebDataset")
    assert hasattr(wds, "PairedShardList")
    assert hasattr(wds, "PairedTarExpander")


@pytest.mark.quick
def test_perfectly_aligned_shards(tmp_path):
    """Test that perfectly aligned shards produce all samples."""
    keys = [f"sample_{i:04d}" for i in range(10)]
    img_samples = [{"__key__": k, "png": np.zeros((3, 3, 3), dtype=np.uint8)} for k in keys]
    emb_samples = [{"__key__": k, "npy": np.ones((16,), dtype=np.float32)} for k in keys]

    _create_tar(f"{tmp_path}/images/shard_0000.tar", img_samples)
    _create_tar(f"{tmp_path}/embeddings/shard_0000.tar", emb_samples)

    modalities = {
        "images": [f"{tmp_path}/images/shard_0000.tar"],
        "embeddings": [f"{tmp_path}/embeddings/shard_0000.tar"],
    }
    ds = MultiModalWebDataset(
        modalities=modalities,
        shardshuffle=False,
        nodesplitter=None,
        workersplitter=None,
        empty_check=False,
    )

    results = list(ds)
    assert len(results) == 10
    result_keys = sorted(r["__key__"] for r in results)
    assert result_keys == sorted(keys)
