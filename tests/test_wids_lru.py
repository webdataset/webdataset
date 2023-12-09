from collections import OrderedDict

from wids.wids_lru import LRUCache


# Test initialization
def test_initialization():
    cache = LRUCache(5)
    assert cache.capacity == 5
    assert isinstance(cache.cache, OrderedDict)


# Test __setitem__ and __getitem__
def test_set_and_get():
    cache = LRUCache(3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    assert cache["a"] == 1  # Existing item retrieval
    assert cache["b"] == 2  # Existing item retrieval
    assert cache["c"] == 3  # Existing item retrieval
    assert cache["d"] is None  # Nonexistent item retrieval

    cache["a"] = 10  # Update an existing item
    assert cache["a"] == 10

    cache["d"] = 4  # Add a new item, exceeding capacity
    assert cache["d"] == 4  # New item retrieval
    assert "b" not in cache  # 'b' should have been evicted due to exceeding capacity


# Test __delitem__
def test_del():
    cache = LRUCache(3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3

    del cache["b"]  # Delete an existing item
    assert cache["b"] is None

    del cache["d"]  # Delete a nonexistent item, should not raise an error


# Test __len__
def test_len():
    cache = LRUCache(3)
    assert len(cache) == 0

    cache["a"] = 1
    assert len(cache) == 1

    cache["b"] = 2
    assert len(cache) == 2

    cache["c"] = 3
    assert len(cache) == 3

    cache["d"] = 4
    assert len(cache) == 3  # Capacity exceeded, so max length should be maintained


# Test __contains__
def test_contains():
    cache = LRUCache(3)
    cache["a"] = 1
    cache["b"] = 2

    assert "a" in cache  # Existing key
    assert "b" in cache  # Existing key
    assert "c" not in cache  # Nonexistent key


# Test cache eviction and release callback
def test_eviction_and_release_callback():
    # Mock release handler for testing
    evicted_keys = []

    def release_handler(key, value):
        evicted_keys.append(key)

    cache = LRUCache(3, release_handler=release_handler)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3

    assert cache["a"] == 1  # Retrieve 'a' to update its position in the cache
    cache["d"] = 4  # Add 'd' to exceed capacity and trigger eviction

    # Check evicted keys
    assert (
        "b" in evicted_keys
    )  # Since 'b' is the least recently used key, it should be evicted
    assert len(evicted_keys) == 1

    # Check cache state after eviction
    assert len(cache) == 3  # Capacity should still be maintained
    assert "b" not in cache  # 'b' should have been evicted

    # Test release handler was called with the correct values
    assert (
        evicted_keys[0] == "b"
    )  # The key that was evicted should be passed to the release handler
