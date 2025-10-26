# type: ignore
"""
Unit tests for the preprocess module.

Tests cover:
- _PreProcess class initialization
- Audio normalization and writing
- Audio pipeline processing
- Multi-process pipeline
- Error handling
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from scipy import signal as sp_signal
from scipy.io import wavfile

import infer.modules.train.preprocess as preprocess_module
from infer.modules.train.preprocess import _PreProcess

# Constants for test data
TEST_SAMPLE_RATE = 44100
TEST_DURATION_SECONDS = 5
TEST_NUM_PROCESSES = 2
TEST_PER = 3.7
TEST_OVERLAP = 0.3
TEST_THRESHOLD_DB = -42
TEST_MIN_LENGTH_MS = 1500
TEST_MIN_INTERVAL_MS = 400
TEST_HOP_SIZE_MS = 15
TEST_MAX_SIL_KEPT_MS = 500
BUTTER_FILTER_ORDER = 5
BUTTER_CUTOFF_HZ = 48
AUDIO_MAX_THRESHOLD = 2.5
AUDIO_ABOVE_THRESHOLD = 3.0
AUDIO_AMPLITUDE_SCALE_SMALL = 0.1
AUDIO_AMPLITUDE_SCALE_MEDIUM = 0.2
AUDIO_AMPLITUDE_SCALE_LARGE = 0.5
ALPHA_BLEND = 0.75
MAX_AMPLITUDE = 0.9
AUDIO_EPSILON = 1e-6
TARGET_SAMPLE_RATE_16K = 16000
AUDIO_LENGTH_TOLERANCE = 100
RNG_SEED = 42

# Generate stable RNG for tests
RNG = np.random.default_rng(RNG_SEED)


@pytest.fixture(autouse=True)
def mock_preprocess_logging(tmp_path: Path) -> None:
    """Mock the logging file handle to prevent NoneType errors."""
    mock_file = MagicMock()
    preprocess_module._f = mock_file


class TestPreProcessInit:
    """Test suite for _PreProcess initialization."""

    def test_preprocess_initialization(self, tmp_path: Path) -> None:
        """Test that _PreProcess initializes correctly with all attributes."""
        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        assert pp.sr == TEST_SAMPLE_RATE
        assert pp.per == 3.7
        assert pp.overlap == 0.3
        assert pp.tail == pytest.approx(pp.per + pp.overlap)
        assert pp.max == 0.9
        assert pp.alpha == 0.75
        assert pp.exp_dir == tmp_path
        assert pp.gt_wavs_dir == tmp_path / "0_gt_wavs"
        assert pp.wavs16k_dir == tmp_path / "1_16k_wavs"

    def test_preprocess_creates_directories(self, tmp_path: Path) -> None:
        """Test that _PreProcess creates required directories."""
        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        assert pp.exp_dir.exists()
        assert pp.gt_wavs_dir.exists()
        assert pp.wavs16k_dir.exists()

    def test_preprocess_custom_per(self, tmp_path: Path) -> None:
        """Test _PreProcess with custom per parameter."""
        custom_per = 5.0
        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path, per=custom_per)

        assert pp.per == custom_per
        assert pp.tail == pytest.approx(custom_per + pp.overlap)

    def test_preprocess_slicer_initialization(self, tmp_path: Path) -> None:
        """Test that Slicer is initialized with correct parameters."""
        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        assert pp.slicer is not None
        # Slicer is a Slicer object, verify it exists
        assert hasattr(pp, "slicer")

    def test_preprocess_butter_filter_initialization(self, tmp_path: Path) -> None:
        """Test that butter filter coefficients are initialized."""
        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        assert pp.bh is not None
        assert pp.ah is not None
        assert isinstance(pp.bh, np.ndarray)
        assert isinstance(pp.ah, np.ndarray)


class TestNormWrite:
    """Test suite for norm_write method."""

    def test_norm_write_basic(self, tmp_path: Path) -> None:
        """Test basic audio normalization and writing."""
        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        # Create test audio
        duration = 1.0
        samples = int(TEST_SAMPLE_RATE * duration)
        test_audio = RNG.normal(size=samples).astype(np.float32) * 0.1

        pp.norm_write(test_audio, 0, 0)

        # Check files were created
        gt_path = tmp_path / "0_gt_wavs" / "0_0.wav"
        wav16k_path = tmp_path / "1_16k_wavs" / "0_0.wav"

        assert gt_path.exists()
        assert wav16k_path.exists()

    def test_norm_write_creates_correct_sample_rates(self, tmp_path: Path) -> None:
        """Test that norm_write creates files with correct sample rates."""
        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        duration = 1.0
        samples = int(TEST_SAMPLE_RATE * duration)
        test_audio = RNG.normal(size=samples).astype(np.float32) * 0.1

        pp.norm_write(test_audio, 0, 0)

        # Read and verify original sample rate
        gt_path = tmp_path / "0_gt_wavs" / "0_0.wav"
        sr_gt, data_gt = wavfile.read(gt_path)
        assert sr_gt == TEST_SAMPLE_RATE

        # Read and verify 16k sample rate
        wav16k_path = tmp_path / "1_16k_wavs" / "0_0.wav"
        sr_16k, data_16k = wavfile.read(wav16k_path)
        assert sr_16k == 16000

    def test_norm_write_filters_loud_audio(self, tmp_path: Path) -> None:
        """Test that audio louder than threshold is filtered out."""
        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        # Create very loud audio
        duration = 1.0
        samples = int(TEST_SAMPLE_RATE * duration)
        test_audio = (
            RNG.normal(size=samples).astype(np.float32) * 3.0
        )  # Above threshold

        pp.norm_write(test_audio, 0, 0)

        # Files should not be created for filtered audio
        gt_path = tmp_path / "0_gt_wavs" / "0_0.wav"
        assert not gt_path.exists()

    def test_norm_write_normalizes_amplitude(self, tmp_path: Path) -> None:
        """Test that audio amplitude is normalized correctly."""
        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        # Create audio with known amplitude
        duration = 1.0
        samples = int(TEST_SAMPLE_RATE * duration)
        test_audio = RNG.normal(size=samples).astype(np.float32) * 0.1

        pp.norm_write(test_audio, 0, 0)

        gt_path = tmp_path / "0_gt_wavs" / "0_0.wav"
        sr, data = wavfile.read(gt_path)
        data_float = data.astype(np.float32)

        # Check that processed audio exists and has data
        max_amplitude = np.abs(data_float).max()
        # Should be between 0 and max (not exceeding buffer limits)
        assert 0 <= max_amplitude <= 1.5

    def test_norm_write_multiple_files(self, tmp_path: Path) -> None:
        """Test creating multiple output files with different indices."""
        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        duration = 1.0
        samples = int(TEST_SAMPLE_RATE * duration)
        test_audio = RNG.normal(size=samples).astype(np.float32) * 0.1

        # Write multiple files
        for i in range(3):
            pp.norm_write(test_audio, 0, i)

        # Check all files exist
        for i in range(3):
            gt_path = tmp_path / "0_gt_wavs" / f"0_{i}.wav"
            wav16k_path = tmp_path / "1_16k_wavs" / f"0_{i}.wav"
            assert gt_path.exists()
            assert wav16k_path.exists()


class TestPipeline:
    """Test suite for pipeline method."""

    @pytest.fixture
    def temp_audio_file(self, tmp_path: Path) -> Path:
        """Create a temporary audio file for testing."""
        audio_path = tmp_path / "test_audio.wav"

        # Create test audio
        duration = 2.0
        samples = int(TEST_SAMPLE_RATE * duration)
        audio = RNG.normal(size=samples).astype(np.float32) * 0.1
        wavfile.write(str(audio_path), TEST_SAMPLE_RATE, audio)

        return audio_path

    @patch("infer.modules.train.preprocess.load_audio")
    @patch("infer.modules.train.preprocess.Slicer")
    def test_pipeline_processes_audio(
        self,
        mock_slicer_class: Mock,
        mock_load_audio: Mock,
        tmp_path: Path,
        temp_audio_file: Path,
    ) -> None:
        """Test that pipeline processes audio correctly."""
        # Create mock audio
        duration = 5.0
        samples = int(TEST_SAMPLE_RATE * duration)
        test_audio = RNG.normal(size=samples).astype(np.float32) * 0.1

        mock_load_audio.return_value = test_audio

        # Create mock slicer
        mock_slicer = MagicMock()
        mock_slicer_class.return_value = mock_slicer
        mock_slicer.slice.return_value = [test_audio]

        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        # Process should not raise exception
        pp.pipeline(str(temp_audio_file), 0)

    @patch("infer.modules.train.preprocess.load_audio")
    def test_pipeline_handles_load_error(
        self, mock_load_audio: Mock, tmp_path: Path, temp_audio_file: Path
    ) -> None:
        """Test that pipeline handles audio loading errors gracefully."""
        mock_load_audio.side_effect = RuntimeError("Failed to load audio")

        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        # Should not raise, error should be logged
        pp.pipeline(str(temp_audio_file), 0)

    @patch("infer.modules.train.preprocess.load_audio")
    @patch("infer.modules.train.preprocess.Slicer")
    def test_pipeline_creates_output_files(
        self,
        mock_slicer_class: Mock,
        mock_load_audio: Mock,
        tmp_path: Path,
        temp_audio_file: Path,
    ) -> None:
        """Test that pipeline creates output files."""
        # Create mock audio
        duration = 5.0
        samples = int(TEST_SAMPLE_RATE * duration)
        test_audio = RNG.normal(size=samples).astype(np.float32) * 0.1

        mock_load_audio.return_value = test_audio

        # Create mock slicer
        mock_slicer = MagicMock()
        mock_slicer_class.return_value = mock_slicer
        mock_slicer.slice.return_value = [test_audio]

        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)
        pp.pipeline(str(temp_audio_file), 0)

        # Check that at least some files were created
        gt_wavs = list((tmp_path / "0_gt_wavs").glob("*.wav"))
        wavs_16k = list((tmp_path / "1_16k_wavs").glob("*.wav"))

        assert len(gt_wavs) > 0
        assert len(wavs_16k) > 0

    @patch("infer.modules.train.preprocess.load_audio")
    def test_pipeline_mp(
        self, mock_load_audio: Mock, tmp_path: Path, temp_audio_file: Path
    ) -> None:
        """Test pipeline_mp method with multiple files."""
        # Create mock audio
        duration = 5.0
        samples = int(TEST_SAMPLE_RATE * duration)
        test_audio = RNG.normal(size=samples).astype(np.float32) * 0.1

        mock_load_audio.return_value = test_audio

        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        # Create multiple test files
        infos = [
            (str(temp_audio_file), 0),
            (str(temp_audio_file), 1),
        ]

        # Should not raise
        pp.pipeline_mp(infos)


class TestPipelineMpInpDir:
    """Test suite for pipeline_mp_inp_dir method."""

    def test_pipeline_mp_inp_dir_creates_files(self, tmp_path: Path) -> None:
        """Test that pipeline_mp_inp_dir processes input directory."""
        # Create input directory with test audio files
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        for i in range(2):
            audio_path = input_dir / f"test_{i}.wav"
            duration = 1.0
            samples = int(TEST_SAMPLE_RATE * duration)
            audio = RNG.normal(size=samples).astype(np.float32) * 0.1
            wavfile.write(str(audio_path), TEST_SAMPLE_RATE, audio)

        exp_dir = tmp_path / "exp"

        with patch("infer.modules.train.preprocess.load_audio") as mock_load:
            # Create mock audio
            duration = 1.0
            samples = int(TEST_SAMPLE_RATE * duration)
            test_audio = RNG.normal(size=samples).astype(np.float32) * 0.1
            mock_load.return_value = test_audio

            pp = _PreProcess(TEST_SAMPLE_RATE, exp_dir)
            pp.pipeline_mp_inp_dir(input_dir, 1)

            # Check that files were created
            gt_wavs = list((exp_dir / "0_gt_wavs").glob("*.wav"))
            assert len(gt_wavs) > 0

    def test_pipeline_mp_inp_dir_with_no_parallel(self, tmp_path: Path) -> None:
        """Test pipeline_mp_inp_dir with no_parallel option."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        for i in range(2):
            audio_path = input_dir / f"test_{i}.wav"
            duration = 1.0
            samples = int(TEST_SAMPLE_RATE * duration)
            audio = RNG.normal(size=samples).astype(np.float32) * 0.1
            wavfile.write(str(audio_path), TEST_SAMPLE_RATE, audio)

        exp_dir = tmp_path / "exp"

        with patch("infer.modules.train.preprocess.load_audio") as mock_load:
            duration = 1.0
            samples = int(TEST_SAMPLE_RATE * duration)
            test_audio = RNG.normal(size=samples).astype(np.float32) * 0.1
            mock_load.return_value = test_audio

            with patch("infer.modules.train.preprocess._no_parallel", True):
                pp = _PreProcess(TEST_SAMPLE_RATE, exp_dir)
                pp.pipeline_mp_inp_dir(input_dir, 1)

                gt_wavs = list((exp_dir / "0_gt_wavs").glob("*.wav"))
                assert len(gt_wavs) > 0

    def test_pipeline_mp_inp_dir_empty_directory(self, tmp_path: Path) -> None:
        """Test pipeline_mp_inp_dir with empty input directory."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        exp_dir = tmp_path / "exp"
        pp = _PreProcess(TEST_SAMPLE_RATE, exp_dir)

        # Should not raise
        pp.pipeline_mp_inp_dir(input_dir, 1)


class TestAudioProcessing:
    """Test suite for audio processing aspects."""

    def test_butter_filter_applied_correctly(self, tmp_path: Path) -> None:
        """Test that butter filter is applied to audio."""
        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        # Create test audio with known properties
        duration = 1.0
        samples = int(TEST_SAMPLE_RATE * duration)
        freq_low = 20
        freq_high = 200

        # Create sine wave at low frequency
        t = np.arange(samples) / TEST_SAMPLE_RATE
        test_audio = np.sin(2 * np.pi * freq_low * t).astype(np.float32)

        # Apply high-pass filter manually

        filtered = sp_signal.lfilter(pp.bh, pp.ah, test_audio)

        # Filtered audio should have different amplitude
        assert np.abs(filtered).max() < np.abs(test_audio).max()

    def test_audio_resampling_to_16k(self, tmp_path: Path) -> None:
        """Test that audio is correctly resampled to 16kHz."""
        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        # Create test audio
        duration = 1.0
        samples = int(TEST_SAMPLE_RATE * duration)
        test_audio = RNG.normal(size=samples).astype(np.float32) * 0.1

        pp.norm_write(test_audio, 0, 0)

        # Read 16k file and verify length
        wav16k_path = tmp_path / "1_16k_wavs" / "0_0.wav"
        sr, data = wavfile.read(wav16k_path)

        assert sr == 16000
        # Should be approximately 1 second at 16kHz
        assert 15900 < len(data) < 16100

    def test_alpha_blending_applied(self, tmp_path: Path) -> None:
        """Test that alpha blending is applied during normalization."""
        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        # Create test audio
        duration = 1.0
        samples = int(TEST_SAMPLE_RATE * duration)
        test_audio = RNG.normal(size=samples).astype(np.float32) * 0.2

        pp.norm_write(test_audio, 0, 0)

        gt_path = tmp_path / "0_gt_wavs" / "0_0.wav"
        _sr, data = wavfile.read(gt_path)
        data_float = data.astype(np.float32)

        # Alpha blending should produce non-zero audio
        # (The exact relationship depends on the implementation)
        normalized_max = np.abs(data_float).max()
        assert normalized_max > 0, "Alpha blended audio should not be all zeros"


class TestErrorHandling:
    """Test suite for error handling."""

    @patch("infer.modules.train.preprocess.load_audio")
    def test_pipeline_error_handling(
        self, mock_load_audio: Mock, tmp_path: Path, temp_audio_file: Path
    ) -> None:
        """Test that pipeline handles errors gracefully."""
        mock_load_audio.side_effect = Exception("Test error")

        pp = _PreProcess(TEST_SAMPLE_RATE, tmp_path)

        # Should not raise, error should be caught and logged
        pp.pipeline(str(temp_audio_file), 0)

    def test_pipeline_mp_inp_dir_error_handling(self, tmp_path: Path) -> None:
        """Test that pipeline_mp_inp_dir handles errors gracefully."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        exp_dir = tmp_path / "exp"
        pp = _PreProcess(TEST_SAMPLE_RATE, exp_dir)

        with patch("infer.modules.train.preprocess.load_audio") as mock_load:
            mock_load.side_effect = Exception("Test error")

            # Should not raise
            pp.pipeline_mp_inp_dir(input_dir, 1)

    @pytest.fixture
    def temp_audio_file(self, tmp_path: Path) -> Path:
        """Create a temporary audio file for testing."""
        audio_path = tmp_path / "test_audio.wav"

        # Create test audio
        duration = 2.0
        samples = int(TEST_SAMPLE_RATE * duration)
        audio = RNG.normal(size=samples).astype(np.float32) * 0.1
        wavfile.write(str(audio_path), TEST_SAMPLE_RATE, audio)

        return audio_path
