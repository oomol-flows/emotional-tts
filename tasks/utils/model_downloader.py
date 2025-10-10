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


def _download_from_huggingface(model_repo: str, model_dir: str) -> None:
    """Download model from HuggingFace"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface-hub package not found. "
            "Please ensure it's installed: pip install huggingface-hub"
        ) from e

    print(">> Model not found, downloading from HuggingFace...")
    print(f"   Repository: {model_repo}")
    print(f"   Destination: {model_dir}")

    start_time = time.time()
    try:
        snapshot_download(
            repo_id=model_repo,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        elapsed = time.time() - start_time
        print(f">> Model download completed successfully in {elapsed:.1f}s")
    except Exception as e:
        raise RuntimeError(f"Failed to download model from HuggingFace: {e}") from e


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
        model_repo: HuggingFace repository ID

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

        # Download the model
        _download_from_huggingface(model_repo, model_dir)

        # Verify config file exists
        if not config_path.exists():
            raise RuntimeError(
                f"Model download completed but config.yaml not found at {config_path}"
            )

    return model_dir
