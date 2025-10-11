"""
Model downloader utility for IndexTTS2 models.
Ensures models are downloaded on first execution with thread-safe locking.
"""

import os
import time
import fcntl
import socket
from pathlib import Path
from contextlib import contextmanager
from urllib.parse import urlparse


@contextmanager
def _file_lock(lock_path: Path):
    """Context manager for file-based locking"""
    lock_file = open(lock_path, 'w')
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
        # Clean up lock file
        lock_path.unlink(missing_ok=True)


def _ping_host(url: str, timeout: float = 2.0) -> float:
    """Ping a host and return latency in milliseconds. Returns float('inf') on failure."""
    try:
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path

        start = time.time()
        sock = socket.create_connection((host, 443), timeout=timeout)
        sock.close()
        latency = (time.time() - start) * 1000

        return latency
    except Exception:
        return float('inf')


def _download_from_huggingface(model_repo: str, model_dir: str, max_retries: int = 3) -> None:
    """Download model from HuggingFace with retry and mirror support"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface-hub package not found. "
            "Please ensure it's installed: pip install huggingface-hub"
        ) from e

    # Define endpoints
    endpoints = [
        ("HuggingFace Official", "https://huggingface.co"),
        ("HuggingFace Mirror (hf-mirror.com)", "https://hf-mirror.com"),
    ]

    print(f">> Model not found, downloading from repository: {model_repo}")
    print(f"   Destination: {model_dir}")

    # Ping all endpoints to find the fastest
    print(">> Testing network latency to endpoints...")
    latencies = []
    for endpoint_name, endpoint_url in endpoints:
        latency = _ping_host(endpoint_url)
        latencies.append((latency, endpoint_name, endpoint_url))
        if latency == float('inf'):
            print(f"   {endpoint_name}: Unreachable")
        else:
            print(f"   {endpoint_name}: {latency:.0f}ms")

    # Sort by latency (fastest first)
    latencies.sort(key=lambda x: x[0])
    sorted_endpoints = [(name, url) for _, name, url in latencies]

    last_error = None

    for endpoint_name, endpoint_url in sorted_endpoints:
        # Set HuggingFace endpoint
        os.environ['HF_ENDPOINT'] = endpoint_url

        for attempt in range(1, max_retries + 1):
            try:
                print(f">> Trying {endpoint_name} (Attempt {attempt}/{max_retries})...")
                start_time = time.time()

                snapshot_download(
                    repo_id=model_repo,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )

                elapsed = time.time() - start_time
                print(f">> Model download completed successfully in {elapsed:.1f}s")
                return

            except Exception as e:
                last_error = e
                print(f"   Failed: {str(e)[:100]}")

                if attempt < max_retries:
                    wait_time = attempt * 2
                    print(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        print(f">> All retries failed for {endpoint_name}, trying next source...")

    raise RuntimeError(f"Failed to download model from all sources. Last error: {last_error}") from last_error


def _download_from_modelscope(model_repo: str, model_dir: str, max_retries: int = 3) -> None:
    """Download model from ModelScope (魔搭)"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print(">> ModelScope SDK not installed, skipping...")
        raise RuntimeError("ModelScope SDK not available. Install with: pip install modelscope")

    print(f">> Trying ModelScope (魔搭)...")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"   Attempt {attempt}/{max_retries}...")
            start_time = time.time()

            # ModelScope snapshot_download returns the cache directory
            downloaded_dir = snapshot_download(
                model_id=model_repo,
                cache_dir=model_dir
            )

            elapsed = time.time() - start_time
            print(f">> Model download completed from ModelScope in {elapsed:.1f}s")
            print(f"   Downloaded to: {downloaded_dir}")
            return

        except Exception as e:
            print(f"   Failed: {str(e)[:100]}")
            if attempt < max_retries:
                wait_time = attempt * 2
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)

    raise RuntimeError("Failed to download from ModelScope after all retries")


def ensure_model_downloaded(
    model_dir: str = "/oomol-driver/oomol-storage/indextts-checkpoints",
    model_repo: str = "IndexTeam/IndexTTS-2"
) -> str:
    """
    Ensure IndexTTS2 model is downloaded and ready to use.
    Downloads the model on first execution if not already present.
    Uses file locking to prevent concurrent downloads.

    Args:
        model_dir: Directory to store the model files
        model_repo: HuggingFace repository ID (also used for ModelScope if available)

    Returns:
        Path to the model directory

    Raises:
        RuntimeError: If model download fails or config.yaml is missing
    """
    model_dir_path = Path(model_dir)
    config_path = model_dir_path / "config.yaml"
    lock_path = model_dir_path / ".download.lock"

    # Fast path: if model exists, return immediately
    if config_path.exists():
        return model_dir

    # Create model directory if needed
    model_dir_path.mkdir(parents=True, exist_ok=True)

    # Use file lock to prevent concurrent downloads
    with _file_lock(lock_path):
        # Double-check after acquiring lock (another process might have downloaded)
        if config_path.exists():
            print(">> Model already downloaded by another process")
            return model_dir

        # Try download from multiple sources
        download_errors = []

        # Test ModelScope availability
        modelscope_available = False
        try:
            import modelscope  # noqa: F401
            modelscope_available = True
        except ImportError:
            pass

        # Ping ModelScope to determine download order
        sources = []

        # Always add HuggingFace
        sources.append(("HuggingFace", _download_from_huggingface))

        # Add ModelScope if available
        if modelscope_available:
            ms_latency = _ping_host("https://modelscope.cn")
            hf_latency = _ping_host("https://huggingface.co")

            print(f">> Source latency comparison:")
            print(f"   HuggingFace: {hf_latency:.0f}ms" if hf_latency != float('inf') else "   HuggingFace: Unreachable")
            print(f"   ModelScope: {ms_latency:.0f}ms" if ms_latency != float('inf') else "   ModelScope: Unreachable")

            # If ModelScope is faster, put it first
            if ms_latency < hf_latency:
                sources.insert(0, ("ModelScope", _download_from_modelscope))
            else:
                sources.append(("ModelScope", _download_from_modelscope))

        # Try each source in order
        for source_name, download_func in sources:
            try:
                print(f">> Attempting download from {source_name}...")
                download_func(model_repo, model_dir)
                break  # Success, exit loop
            except Exception as e:
                download_errors.append(f"{source_name}: {e}")
                print(f">> {source_name} download failed")

        # Verify config file exists
        if not config_path.exists():
            error_msg = "\n".join(download_errors)
            raise RuntimeError(
                f"Model download failed from all sources.\n{error_msg}\n"
                f"Config file not found at {config_path}"
            )

        print(f">> Model verified successfully at {model_dir}")

    return model_dir
