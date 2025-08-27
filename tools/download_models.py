import os
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Base URL for model downloads
RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"

# Determine the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

def dl_model(link: str, model_name: str, download_dir: Path) -> Optional[Path]:
    url = f"{link}{model_name}"
    target_path = download_dir / model_name
    
    # Check if the file already exists
    if target_path.exists():
        print(f"Skipping download: {model_name} already exists.")
        return target_path

    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Downloading {model_name}...")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            
            with open(target_path, "wb") as f:
                with tqdm(
                    total=total_size, 
                    unit="B", 
                    unit_scale=True, 
                    desc=model_name,
                    miniters=1,
                ) as progress_bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        progress_bar.update(len(chunk))

        print(f"Successfully downloaded {model_name} to {target_path}")
        return target_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {model_name}: {e}")
        if target_path.exists():
            os.remove(target_path)
        return None

def main():
    # Define directory paths using Path objects
    assets_dir = BASE_DIR / "assets"
    hubert_dir = assets_dir / "hubert"
    rmvpe_dir = assets_dir / "rmvpe"
    pretrained_dir = assets_dir / "pretrained"
    pretrained_v2_dir = assets_dir / "pretrained_v2"

    # List of models to download
    core_models = {
        "hubert_base.pt": hubert_dir,
        "rmvpe.pt": rmvpe_dir,
    }
    
    pretrained_models = [
        "D32k.pth", "D40k.pth", "D48k.pth", "G32k.pth", "G40k.pth", "G48k.pth",
        "f0D32k.pth", "f0D40k.pth", "f0D48k.pth", "f0G32k.pth", "f0G40k.pth", "f0G48k.pth",
    ]

    # Download core models
    for model, path in core_models.items():
        dl_model(RVC_DOWNLOAD_LINK, model, path)

    # Download pretrained models (v1)
    for model in pretrained_models:
        dl_model(RVC_DOWNLOAD_LINK + "pretrained/", model, pretrained_dir)
        
    # Download pretrained models v2
    for model in pretrained_models:
        dl_model(RVC_DOWNLOAD_LINK + "pretrained_v2/", model, pretrained_v2_dir)

    print("\nAll model downloads complete!")

if __name__ == "__main__":
    main()