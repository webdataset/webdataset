"""Tests for security features and enforce_security flag."""

import os
import pickle
import tempfile

import pytest

import webdataset as wds
from webdataset.autodecode import torch_loads, unpickle_loads
from webdataset.gopen import gopen, rewrite_url


class TestEnforceSecurity:
    """Test cases for the enforce_security flag."""

    def setup_method(self):
        """Reset enforce_security before each test."""
        wds.utils.enforce_security = False

    def teardown_method(self):
        """Reset enforce_security after each test."""
        wds.utils.enforce_security = False

    def test_enforce_security_default_false(self):
        """Test that enforce_security defaults to False."""
        assert wds.utils.enforce_security is False

    def test_enforce_security_can_be_set(self):
        """Test that enforce_security can be modified."""
        wds.utils.enforce_security = True
        assert wds.utils.enforce_security is True

        wds.utils.enforce_security = False
        assert wds.utils.enforce_security is False

    def test_pickle_loads_allowed_when_security_disabled(self):
        """Test that pickle loading works when enforce_security is False."""
        wds.utils.enforce_security = False
        test_data = {"test": "data", "number": 42}
        pickled_data = pickle.dumps(test_data)

        # Should work without error
        result = unpickle_loads(pickled_data)
        assert result == test_data

    def test_pickle_loads_blocked_when_security_enabled(self):
        """Test that pickle loading is blocked when enforce_security is True."""
        wds.utils.enforce_security = True
        test_data = {"test": "data", "number": 42}
        pickled_data = pickle.dumps(test_data)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Unpickling is not allowed for security reasons when enforce_security is set"):
            unpickle_loads(pickled_data)

    def test_torch_loads_allowed_when_security_disabled(self):
        """Test that torch loading works when enforce_security is False."""
        pytest.importorskip("torch")
        import io

        import torch

        wds.utils.enforce_security = False
        test_tensor = torch.tensor([1, 2, 3, 4])

        # Create torch data
        buffer = io.BytesIO()
        torch.save(test_tensor, buffer)
        torch_data = buffer.getvalue()

        # Should work without error
        result = torch_loads(torch_data)
        assert torch.equal(result, test_tensor)

    def test_torch_loads_blocked_when_security_enabled(self):
        """Test that torch loading is blocked when enforce_security is True."""
        pytest.importorskip("torch")
        import io

        import torch

        wds.utils.enforce_security = True
        test_tensor = torch.tensor([1, 2, 3, 4])

        # Create torch data
        buffer = io.BytesIO()
        torch.save(test_tensor, buffer)
        torch_data = buffer.getvalue()

        # Should raise ValueError
        with pytest.raises(ValueError, match="torch.loads is not allowed for security reasons when enforce_security is set"):
            torch_loads(torch_data)

    def test_local_file_access_allowed_when_security_disabled(self):
        """Test that local file access works when enforce_security is False."""
        wds.utils.enforce_security = False

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            # Should work without error
            with gopen(temp_path, "rb") as stream:
                content = stream.read()
                assert content == b"test content"
        finally:
            os.unlink(temp_path)

    def test_local_file_access_blocked_when_security_enabled(self):
        """Test that local file access is blocked when enforce_security is True."""
        wds.utils.enforce_security = True

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            # Should raise ValueError
            with pytest.raises(ValueError, match="gopen: unsafe_gopen is False, cannot open local files"):
                gopen(temp_path, "rb")
        finally:
            os.unlink(temp_path)

    def test_file_url_access_allowed_when_security_disabled(self):
        """Test that file:// URL access works when enforce_security is False."""
        wds.utils.enforce_security = False

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            file_url = f"file://{temp_path}"
            # Should work without error
            with gopen(file_url, "rb") as stream:
                content = stream.read()
                assert content == b"test content"
        finally:
            os.unlink(temp_path)

    def test_file_url_access_blocked_when_security_enabled(self):
        """Test that file:// URL access is blocked when enforce_security is True."""
        wds.utils.enforce_security = True

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            file_url = f"file://{temp_path}"
            # Should raise ValueError
            with pytest.raises(ValueError, match="gopen: unsafe_gopen is False, cannot open local files"):
                gopen(file_url, "rb")
        finally:
            os.unlink(temp_path)

    def test_pipe_url_allowed_when_security_disabled(self):
        """Test that pipe: URLs work when enforce_security is False."""
        wds.utils.enforce_security = False

        # Should work without error
        with gopen("pipe:echo 'hello world'", "rb") as stream:
            content = stream.read()
            assert content.strip() == b"hello world"

    def test_pipe_url_blocked_when_security_enabled(self):
        """Test that pipe: URLs are blocked when enforce_security is True."""
        wds.utils.enforce_security = True

        # Should raise ValueError
        with pytest.raises(ValueError, match="gopen_pipe: unsafe_gopen is False, cannot open pipe URLs"):
            gopen("pipe:echo 'hello world'", "rb")

    def test_url_rewriting_allowed_when_security_disabled(self):
        """Test that URL rewriting works when enforce_security is False."""
        wds.utils.enforce_security = False

        # Set up environment variable for rewriting
        original_env = os.environ.get("GOPEN_REWRITE")
        os.environ["GOPEN_REWRITE"] = "http://example.com/=http://redirected.com/"

        try:
            # Should work without error
            result = rewrite_url("http://example.com/test.txt")
            assert result == "http://redirected.com/test.txt"
        finally:
            if original_env is None:
                os.environ.pop("GOPEN_REWRITE", None)
            else:
                os.environ["GOPEN_REWRITE"] = original_env

    def test_url_rewriting_blocked_when_security_enabled(self):
        """Test that URL rewriting is blocked when enforce_security is True."""
        wds.utils.enforce_security = True

        # Set up environment variable for rewriting
        original_env = os.environ.get("GOPEN_REWRITE")
        os.environ["GOPEN_REWRITE"] = "http://example.com/=http://redirected.com/"

        try:
            # Should raise ValueError
            with pytest.raises(ValueError, match="rewrite_url: unsafe_gopen is False, cannot rewrite URLs using environment variables"):
                rewrite_url("http://example.com/test.txt")
        finally:
            if original_env is None:
                os.environ.pop("GOPEN_REWRITE", None)
            else:
                os.environ["GOPEN_REWRITE"] = original_env

    def test_safe_urls_still_work_when_security_enabled(self):
        """Test that safe URLs (http, https) still work when enforce_security is True."""
        wds.utils.enforce_security = True

        # These should work even with security enabled (though they might fail due to network)
        # We're just testing that they don't get blocked by the security check
        try:
            # This might fail due to network issues, but shouldn't fail due to security
            gopen("http://httpbin.org/status/200", "rb")
        except Exception as e:
            # Network errors are okay, but security errors are not
            assert "unsafe_gopen is False" not in str(e)

    def test_stdin_stdout_still_work_when_security_enabled(self):
        """Test that stdin/stdout ('-') still work when enforce_security is True."""
        wds.utils.enforce_security = True

        # These should work even with security enabled
        stdin_stream = gopen("-", "rb")
        assert stdin_stream is not None

        stdout_stream = gopen("-", "wb")
        assert stdout_stream is not None

    def test_autodecode_integration_with_security(self):
        """Test that autodecode respects the enforce_security flag."""
        pytest.importorskip("torch")
        import io

        import torch

        # Create test data
        test_tensor = torch.tensor([1, 2, 3])
        buffer = io.BytesIO()
        torch.save(test_tensor, buffer)
        torch_data = buffer.getvalue()

        test_dict = {"key": "value"}
        pickle_data = pickle.dumps(test_dict)

        # Test with security disabled
        wds.utils.enforce_security = False

        # These should work
        from webdataset.autodecode import decoders
        assert decoders["pth"](torch_data) is not None
        assert decoders["pkl"](pickle_data) == test_dict

        # Test with security enabled
        wds.utils.enforce_security = True

        # These should fail
        with pytest.raises(ValueError, match="torch.loads is not allowed"):
            decoders["pth"](torch_data)

        with pytest.raises(ValueError, match="Unpickling is not allowed"):
            decoders["pkl"](pickle_data)


if __name__ == "__main__":
    pytest.main([__file__])
