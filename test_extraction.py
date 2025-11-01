#!/usr/bin/env python3
"""End-to-end test for F0 and feature extraction with spawn multiprocessing."""

import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import shared
from tabs.train.extract_pitch import (
    F0FeatureExtractor,
    FeatureExtractor,
    _extract_f0_worker,
)


def create_test_audio(path: Path, duration: float = 2.0, sr: int = 16000) -> None:
    """Create a synthetic audio file for testing."""
    t = np.linspace(0, duration, int(sr * duration))
    # Create a simple sine wave (440 Hz A4 note)
    frequency = 440
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    sf.write(path, audio, sr)
    print(f"✓ Created test audio: {path}")


def test_f0_extraction() -> None:
    """Test F0 extraction with different methods."""
    print("\n" + "=" * 60)
    print("TEST 1: F0 Extraction")
    print("=" * 60)

    with TemporaryDirectory() as tmpdir_ctx:
        tmpdir = Path(tmpdir_ctx)

        # Create test audio files
        wav_dir = tmpdir / "wavs"
        wav_dir.mkdir()

        for i in range(3):
            create_test_audio(wav_dir / f"test_{i}.wav")

        # Create output directories
        f0_dir = tmpdir / "f0"
        f0_nsf_dir = tmpdir / "f0_nsf"
        f0_dir.mkdir()
        f0_nsf_dir.mkdir()

        # Create log file
        log_file = tmpdir / "extraction.log"
        log_file.touch()

        # Prepare paths for extraction
        paths = []
        for wav_file in sorted(wav_dir.glob("*.wav")):
            name = wav_file.name.replace(".wav", ".npy")
            paths.append((wav_file, f0_dir / name, f0_nsf_dir / name))

        # Test F0 methods
        methods = ["pm", "dio"]  # Skip harvest and rmvpe for speed

        for method_name in methods:
            print(f"\n→ Testing F0 method: {method_name}")
            try:
                f0_extractor = F0FeatureExtractor(log_file)
                # Cast to Literal type for type checking
                method = method_name  # type: ignore
                f0_extractor.extract_f0_batch(
                    paths, method, is_half=False, device="cpu"
                )

                # Verify output files were created
                created_files = list(f0_dir.glob("*.npy")) + list(
                    f0_nsf_dir.glob("*.npy")
                )
                if len(created_files) >= 6:  # 3 files * 2 types
                    print(
                        f"  ✓ {method_name}: Successfully created"
                        f" {len(created_files)} output files"
                    )
                else:
                    print(
                        f"  ⚠ {method_name}: Expected 6+ files,"
                        f" got {len(created_files)}"
                    )
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                traceback.print_exc()


def test_feature_extraction() -> None:
    """Test HuBERT feature extraction."""
    print("\n" + "=" * 60)
    print("TEST 2: Feature Extraction")
    print("=" * 60)

    # Check if HuBERT model exists
    hubert_path = Path("assets/hubert/hubert_base.pt")
    if not hubert_path.exists():
        print(f"⚠ HuBERT model not found at {hubert_path}")
        print("  Skipping feature extraction test")
        return

    with TemporaryDirectory() as tmpdir_ctx:
        tmpdir = Path(tmpdir_ctx)

        # Create test directory structure
        exp_dir = tmpdir / "experiment"
        wav_16k_dir = exp_dir / shared.WAVS_16K_DIR_NAME
        wav_16k_dir.mkdir(parents=True)

        # Create test audio files
        for i in range(3):
            create_test_audio(wav_16k_dir / f"test_{i}.wav")

        # Create output directories
        feature_dir = exp_dir / shared.FEATURE_DIR_NAME_V2
        feature_dir.mkdir(parents=True)

        # Create log file
        log_file = tmpdir / "extraction.log"
        log_file.touch()

        print("\n→ Testing feature extraction with device: cpu")
        try:
            feature_extractor = FeatureExtractor(exp_dir, log_file)

            if not feature_extractor.load_model(hubert_path, "cpu", is_half=False):
                print("  ✗ Failed to load HuBERT model")
                return

            file_list = sorted(wav_16k_dir.glob("*.wav"))
            feature_extractor.extract_features_batch(
                file_list, "cpu", "v2", is_half=False
            )

            # Verify output files were created
            created_files = list(feature_dir.glob("*.npy"))
            if len(created_files) >= 3:
                print(
                    f"  ✓ Feature extraction: Successfully created"
                    f" {len(created_files)} output files"
                )
            else:
                print(
                    f"  ⚠ Feature extraction: Expected 3+ files,"
                    f" got {len(created_files)}"
                )
        except Exception as e:
            print(f"  ✗ Feature extraction failed: {e}")
            traceback.print_exc()


def test_worker_functions() -> None:
    """Test worker functions for multiprocessing compatibility."""
    print("\n" + "=" * 60)
    print("TEST 3: Worker Function Compatibility")
    print("=" * 60)

    with TemporaryDirectory() as tmpdir_ctx:
        tmpdir = Path(tmpdir_ctx)

        # Create test audio files
        wav_dir = tmpdir / "wavs"
        wav_dir.mkdir()

        for i in range(2):
            create_test_audio(wav_dir / f"test_{i}.wav")

        # Create output directories
        f0_dir = tmpdir / "f0"
        f0_nsf_dir = tmpdir / "f0_nsf"
        f0_dir.mkdir()
        f0_nsf_dir.mkdir()

        # Create log file
        log_file = tmpdir / "extraction.log"
        log_file.touch()

        # Prepare paths
        paths = []
        for wav_file in sorted(wav_dir.glob("*.wav")):
            name = wav_file.name.replace(".wav", ".npy")
            paths.append((wav_file, f0_dir / name, f0_nsf_dir / name))

        print("\n→ Testing _extract_f0_worker function")
        try:
            # This simulates what happens in a spawned process
            _extract_f0_worker(paths, log_file, "pm", False, "cpu")

            created_files = list(f0_dir.glob("*.npy")) + list(f0_nsf_dir.glob("*.npy"))
            if len(created_files) >= 4:
                print("  ✓ _extract_f0_worker: Successfully processed files")
            else:
                print(
                    f"  ⚠ _extract_f0_worker: Expected 4+ files,"
                    f" got {len(created_files)}"
                )
        except Exception as e:
            print(f"  ✗ _extract_f0_worker failed: {e}")
            traceback.print_exc()


def test_imports() -> bool:
    """Test that all required modules can be imported."""
    print("\n" + "=" * 60)
    print("TEST 0: Import Verification")
    print("=" * 60)

    imports_to_test = [
        ("torch", "PyTorch"),
        ("fairseq", "Fairseq"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("parselmouth", "Parselmouth"),
        ("pyworld", "PyWorld"),
    ]

    all_ok = True
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            all_ok = False

    # Test CUDA availability
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  ✓ PyTorch device: {device}")
        if device == "cuda":
            print(f"    - CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"  ⚠ PyTorch device check: {e}")

    return all_ok


def main() -> int:
    """Run all tests."""
    print("\n")
    print("█" * 60)
    print("  END-TO-END EXTRACTION TEST SUITE")
    print("█" * 60)

    # Test imports first
    if not test_imports():
        print("\n✗ Some dependencies are missing. Cannot proceed with tests.")
        return 1

    try:
        # Run tests
        test_f0_extraction()
        test_feature_extraction()
        test_worker_functions()

        print("\n" + "=" * 60)
        print("TESTS COMPLETED")
        print("=" * 60)
        print("\nCheck the output above for any issues marked with ✗ or ⚠")
        return 0
    except KeyboardInterrupt:
        print("\n\n✗ Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
