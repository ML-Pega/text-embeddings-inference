"""
BGE-M3 sparse embedding files downloader.

Automatically downloads required files for sparse embedding functionality:
- sparse_linear.pt: Required for sparse (lexical) embeddings
- colbert_linear.pt: Required for ColBERT embeddings
"""
import os
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# BGE-M3 model info
MODEL_ID = "BAAI/bge-m3"
HF_BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main"

# Required files for sparse embedding
SPARSE_REQUIRED_FILES = [
    "sparse_linear.pt",
    "colbert_linear.pt",
]


def download_file(url: str, output_path: Path, timeout: int = 300) -> bool:
    """
    Download a file from URL to output path.
    
    Args:
        url: Source URL to download from
        output_path: Destination path for the downloaded file
        timeout: Download timeout in seconds
    
    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"Downloading: {url}")
        
        # Get proxy from environment
        http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
        https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        
        # Setup proxy handler if proxy is configured
        if http_proxy or https_proxy:
            proxy_handler = urllib.request.ProxyHandler({
                'http': http_proxy,
                'https': https_proxy or http_proxy,
            })
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)
        
        # Create a request with user agent to avoid 403
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; TEI-BGE-M3-Downloader/1.0)'}
        )
        
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total_size = response.getheader('Content-Length')
            if total_size:
                total_size = int(total_size)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                block_size = 8192
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    
                    if total_size and downloaded % (1024 * 1024) < block_size:
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        logger.info(f"  Progress: {mb_downloaded:.1f}/{mb_total:.1f} MB")
        
        logger.info(f"Successfully downloaded: {output_path}")
        return True
        
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP Error downloading {url}: {e.code} {e.reason}")
        return False
    except urllib.error.URLError as e:
        logger.error(f"URL Error downloading {url}: {e.reason}")
        return False
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def ensure_sparse_files(model_path: Path) -> Tuple[bool, List[str]]:
    """
    Ensure all required sparse embedding files exist in the model directory.
    Downloads missing files from HuggingFace if necessary.
    
    Args:
        model_path: Path to the model directory (snapshot directory)
    
    Returns:
        Tuple of (success, list of downloaded files)
    """
    model_path = Path(model_path)
    downloaded = []
    all_present = True
    
    logger.info(f"Checking sparse embedding files in: {model_path}")
    
    for filename in SPARSE_REQUIRED_FILES:
        file_path = model_path / filename
        
        if file_path.exists():
            logger.info(f"Found existing file: {filename}")
            continue
        
        logger.info(f"Missing file: {filename}, downloading...")
        url = f"{HF_BASE_URL}/{filename}"
        
        if download_file(url, file_path):
            downloaded.append(filename)
        else:
            logger.error(f"Failed to download: {filename}")
            all_present = False
    
    if downloaded:
        logger.info(f"Downloaded {len(downloaded)} files: {', '.join(downloaded)}")
    
    if all_present:
        logger.info("All sparse embedding files are ready")
    else:
        logger.warning("Some sparse embedding files could not be downloaded")
    
    return all_present, downloaded


def find_snapshot_dir(model_path_or_id: str) -> Optional[Path]:
    """
    Find the actual snapshot directory for a model.
    
    The model_path can be:
    - A direct path to the snapshot directory
    - A HuggingFace cache path like /data/models--BAAI--bge-m3
    - A model ID like "BAAI/bge-m3"
    
    Args:
        model_path_or_id: Model path or ID
    
    Returns:
        Path to the snapshot directory, or None if not found
    """
    path = Path(model_path_or_id)
    
    # Check if it's already a valid snapshot directory
    if path.is_dir():
        # Check if sparse files or model files exist here
        if (path / "config.json").exists() or (path / "pytorch_model.bin").exists():
            return path
        
        # Check if it's a cache directory with snapshots
        snapshots_dir = path / "snapshots"
        if snapshots_dir.exists():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                return snapshots[0]
    
    # Try to find in HuggingFace cache
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/data"),  # Docker volume mount
        Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
    ]
    
    model_dir_name = f"models--{model_path_or_id.replace('/', '--')}"
    
    for cache_dir in cache_dirs:
        if cache_dir is None or not cache_dir.exists():
            continue
        
        model_dir = cache_dir / model_dir_name
        if model_dir.exists():
            snapshots_dir = model_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    return snapshots[0]
    
    return None


def setup_bge_m3_sparse(model_path: str) -> bool:
    """
    Main entry point to setup BGE-M3 sparse embedding support.
    
    This function:
    1. Finds the model snapshot directory
    2. Downloads missing sparse embedding files
    
    Args:
        model_path: Path to the model directory or model ID
    
    Returns:
        True if setup successful, False otherwise
    """
    logger.info(f"Setting up BGE-M3 sparse embedding support for: {model_path}")
    
    # Find the actual snapshot directory
    snapshot_dir = find_snapshot_dir(model_path)
    
    if snapshot_dir is None:
        logger.error(f"Could not find model directory for: {model_path}")
        return False
    
    logger.info(f"Found snapshot directory: {snapshot_dir}")
    
    # Ensure sparse files exist
    success, downloaded = ensure_sparse_files(snapshot_dir)
    
    return success
