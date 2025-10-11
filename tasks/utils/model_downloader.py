"""
Model downloader utility for IndexTTS2 models.
Ensures models are downloaded on first execution with thread-safe locking.
"""

import time
import fcntl
from pathlib import Path
from contextlib import contextmanager


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




def _download_from_huggingface(model_repo: str, model_dir: str, max_retries: int = 3) -> None:
    """Download model from HuggingFace with retry and mirror support"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface-hub package not found. "
            "Please ensure it's installed: pip install huggingface-hub"
        ) from e

    print(f">> Downloading from HuggingFace: {model_repo}")
    print(f"   Destination: {model_dir}")

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f">> Attempt {attempt}/{max_retries}...")
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

    raise RuntimeError(f"Failed to download model from HuggingFace: {last_error}") from last_error




def ensure_model_downloaded(
    model_dir: str = "/oomol-driver/oomol-storage/indextts-checkpoints",
    model_repo: str = "IndexTeam/IndexTTS-2",
    model_source: str = "huggingface"
) -> str:
    """
    Ensure IndexTTS2 model is downloaded and ready to use.
    Downloads the model on first execution if not already present.
    Uses file locking to prevent concurrent downloads.

    Args:
        model_dir: Directory to store the model files
        model_repo: HuggingFace repository ID
        model_source: Download source - only "huggingface" is supported

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

        # Download from HuggingFace
        print(">> Downloading from HuggingFace...")

        try:
            _download_from_huggingface(model_repo, model_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to download model from HuggingFace: {e}") from e

        # Verify config file exists
        if not config_path.exists():
            raise RuntimeError(
                f"Model download completed but config file not found at {config_path}"
            )

        print(f">> Model verified successfully at {model_dir}")

    return model_dir
