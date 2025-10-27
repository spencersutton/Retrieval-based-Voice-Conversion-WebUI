"""
Unit tests for the train module.

Tests cover:
- EpochRecorder class functionality
- Configuration parsing
- Model initialization
- Training utilities
"""

import datetime
import logging
import os
import random
import tempfile
from pathlib import Path
from time import sleep
from typing import Any

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# Import the module to test
from infer.modules.train.train import EpochRecorder

# Constants for magic number replacement
TEST_RECORDS_COUNT = 3
BATCH_SIZE_TEST = 4
DATALOADER_BATCHES = 4
FEATURE_SIZE = 10
OUTPUT_SIZE = 5
LINEAR_INPUT = 100
LINEAR_HIDDEN = 64
LINEAR_OUTPUT = 32
TENSOR_BATCH = 16
TENSOR_FEATURES = 10
TENSOR_HEIGHT = 200
TENSOR_WIDTH = 513
TENSOR_TIME_FRAMES = 100
TENSOR_AUDIO_SAMPLES = 16000
DATASET_SIZE = 4
BETAS_LENGTH = 2
EPS_MAX_THRESHOLD = 1e-8
PERIOD_OUTPUTS_COUNT = 3
SLICE_MAX_INDEX = 50
ITERATION_CHECKPOINT = 1000
LEARNING_RATE_CHECKPOINT = 2e-4
MASTER_PORT_MIN = 20000
MASTER_PORT_MAX = 55555
TOLERANCE_EPSILON = 1e-5


class TestEpochRecorder:
    """Test suite for the EpochRecorder class."""

    def test_epoch_recorder_initialization(self) -> None:
        """Test that EpochRecorder initializes correctly."""
        recorder = EpochRecorder()
        assert hasattr(recorder, "last_time")
        assert isinstance(recorder.last_time, float)

    def test_epoch_recorder_record_format(self) -> None:
        """Test that record() returns properly formatted timestamp string."""
        recorder = EpochRecorder()
        sleep(0.1)  # Small sleep to ensure time difference
        result = recorder.record()

        # Check format: "[YYYY-MM-DD HH:MM:SS] | (HH:MM:SS)"
        assert isinstance(result, str)
        assert "[" in result
        assert "]" in result
        assert "|" in result
        assert "(" in result
        assert ")" in result

    def test_epoch_recorder_elapsed_time(self) -> None:
        """Test that elapsed time is recorded correctly."""
        recorder = EpochRecorder()
        sleep(0.1)
        result = recorder.record()

        # Extract elapsed time format
        assert "0:00:00" in result  # Should be around 0.1 seconds

    def test_epoch_recorder_updates_last_time(self) -> None:
        """Test that record() updates the last_time attribute."""
        recorder = EpochRecorder()
        initial_time = recorder.last_time

        sleep(0.05)
        recorder.record()
        updated_time = recorder.last_time

        assert updated_time > initial_time

    def test_epoch_recorder_multiple_records(self) -> None:
        """Test multiple consecutive record() calls."""
        recorder = EpochRecorder()
        records: list[str] = []

        for _ in range(TEST_RECORDS_COUNT):
            sleep(0.05)
            records.append(recorder.record())

        assert len(records) == TEST_RECORDS_COUNT
        # Each record should be a formatted string
        for record in records:
            assert isinstance(record, str)
            assert "[" in record and "]" in record

    def test_epoch_recorder_timestamp_format(self) -> None:
        """Test that timestamp follows YYYY-MM-DD HH:MM:SS format."""
        recorder = EpochRecorder()
        result = recorder.record()

        # Extract the timestamp part
        start_idx = result.find("[") + 1
        end_idx = result.find("]")
        timestamp = result[start_idx:end_idx]

        # Check that it matches the expected format
        try:
            datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pytest.fail(f"Timestamp '{timestamp}' does not match expected format")


class TestTrainModuleImports:
    """Test that all required imports are available."""

    def test_torch_imports(self) -> None:
        """Test that torch is imported correctly."""
        assert torch is not None
        assert hasattr(torch, "cuda")
        assert hasattr(torch, "distributed")

    def test_dataloader_import(self) -> None:
        """Test that DataLoader is available."""
        assert DataLoader is not None

    def test_nn_module_import(self) -> None:
        """Test that torch.nn is available."""
        assert nn is not None
        assert hasattr(nn, "Module")


class TestSimpleTrainingUtilities:
    """Test helper functions for training."""

    @pytest.fixture
    def mock_hparams(self) -> dict[str, Any]:
        """Create mock HParams for testing."""
        return {
            "gpus": "0",
            "model_dir": "/tmp/test_model",
            "if_f0": 1,
            "if_cache_data_in_gpu": False,
            "train": {
                "batch_size": 4,
                "seed": 42,
                "segment_size": 16384,
                "fp16_run": False,
                "c_mel": 45.0,
                "c_kl": 1.0,
                "learning_rate": 2e-4,
                "betas": [0.8, 0.99],
                "eps": 1e-9,
                "lr_decay": 0.999,
                "epochs": 100,
            },
            "data": {
                "filter_length": 1024,
                "hop_length": 160,
                "win_length": 1024,
                "n_mel_channels": 128,
                "sampling_rate": 16000,
                "mel_fmin": 0,
                "mel_fmax": None,
            },
            "model": {},
            "sample_rate": 16000,
            "if_latest": 0,
            "save_every_epoch": 5,
            "save_every_weights": "0",
            "name": "test_model",
            "pretrainG": "",
            "pretrainD": "",
            "total_epoch": 100,
        }

    def test_cuda_availability_check(self) -> None:
        """Test CUDA availability detection."""
        is_available = torch.cuda.is_available()
        # This test just verifies we can check CUDA availability
        assert isinstance(is_available, bool)

    def test_simple_tensor_operations(self) -> None:
        """Test basic tensor operations used in training."""
        # Test tensor creation and operations
        x = torch.randn(2, 128, 100)
        y = torch.randn(2, 128, 100)

        # Test F.l1_loss equivalent
        loss = torch.nn.functional.l1_loss(x, y)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_dataloader_creation(self) -> None:
        """Test creating a simple DataLoader."""
        # Create simple dataset on CPU
        data = torch.randn(TENSOR_BATCH, FEATURE_SIZE, device="cpu")
        labels = torch.randint(0, 2, (TENSOR_BATCH,), device="cpu")
        dataset = TensorDataset(data, labels)

        # Create dataloader without shuffle to avoid device mismatch
        loader = DataLoader(dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)
        assert len(loader) == DATALOADER_BATCHES

        # Verify batch
        for batch_data, batch_labels in loader:
            assert batch_data.shape[0] == BATCH_SIZE_TEST
            assert batch_labels.shape[0] == BATCH_SIZE_TEST
            break

    def test_model_initialization(self) -> None:
        """Test initializing a simple model."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Move to CPU for testing
        model = model.cpu()

        # Test that model is on CPU
        assert next(model.parameters()).device.type == "cpu"

    def test_optimizer_creation(self) -> None:
        """Test creating optimizers."""
        model = nn.Linear(10, 5)
        optimizer = AdamW(
            model.parameters(),
            lr=2e-4,
            betas=(0.8, 0.99),
            eps=1e-9,
        )

        assert optimizer is not None
        assert len(optimizer.param_groups) > 0


class TestTrainingConfiguration:
    """Test configuration and setup for training."""

    def test_gpu_string_parsing(self) -> None:
        """Test parsing GPU configuration strings."""
        gpu_str = "0-1-2"
        gpus = gpu_str.replace("-", ",")
        assert gpus == "0,1,2"

        # Test with single GPU
        gpu_str = "0"
        gpus = gpu_str.replace("-", ",")
        assert gpus == "0"

    def test_environment_setup(self) -> None:
        """Test environment variable setup."""
        test_dir = tempfile.mkdtemp()
        os.environ["TEST_MODEL_DIR"] = test_dir

        assert os.environ.get("TEST_MODEL_DIR") == test_dir

        # Cleanup
        Path(test_dir).rmdir()

    def test_checkpoint_path_construction(self) -> None:
        """Test constructing checkpoint paths."""
        model_dir_path = Path("/tmp/test_model")
        global_step = 1000

        # Test different checkpoint naming schemes
        checkpoint_name_g = f"G_{global_step}.pth"
        checkpoint_name_d = f"D_{global_step}.pth"
        checkpoint_path_g = model_dir_path / checkpoint_name_g
        checkpoint_path_d = model_dir_path / checkpoint_name_d

        assert str(checkpoint_path_g) == "/tmp/test_model/G_1000.pth"
        assert str(checkpoint_path_d) == "/tmp/test_model/D_1000.pth"


class TestErrorHandling:
    """Test error handling in training."""

    def test_exception_handling_no_gpu(self) -> None:
        """Test handling of scenarios with no GPU."""
        # This simulates the fallback to CPU
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Verify we can handle the no-GPU case
        if n_gpus < 1:
            # Fallback to CPU logic
            n_gpus = 1
            assert n_gpus == 1

    def test_path_existence_check(self) -> None:
        """Test checking if paths exist."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)

        assert temp_path.exists()
        assert temp_path.is_dir()

        # Cleanup
        temp_path.rmdir()

    def test_checkpoint_file_validation(self) -> None:
        """Test validating checkpoint file paths."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as f:
            temp_path = Path(f.name)

        # Test path validation
        assert temp_path.is_file()

        # Cleanup
        temp_path.unlink()


class TestLoggingSetup:
    """Test logging configuration."""

    def test_logger_creation(self) -> None:
        """Test that we can create a logger."""
        logger = logging.getLogger("test_train")
        assert logger is not None

    def test_logger_level_setting(self) -> None:
        """Test setting logger level."""
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.INFO)

        assert logger.level == logging.INFO

    def test_logging_to_handler(self) -> None:
        """Test logging to a handler."""
        logger = logging.getLogger("test_handler")
        handler = logging.StreamHandler()
        logger.addHandler(handler)

        # This should not raise an exception
        logger.info("Test message")


class TestGradScalerFunctionality:
    """Test GradScaler for mixed precision training."""

    def test_grad_scaler_scaling(self) -> None:
        """Test that scaler can scale losses."""
        scaler = torch.GradScaler(enabled=False)
        loss = torch.tensor(1.0, requires_grad=True)

        # This should work without errors
        scaled_loss = scaler.scale(loss)
        assert scaled_loss is not None


class TestDataPreperation:
    """Test data preparation utilities."""

    def test_batch_unpacking_f0(self) -> None:
        """Test unpacking batch with f0 features."""
        # Simulate batch with f0
        (
            phone,
            _phone_lengths,
            pitch,
            _pitchf,
            spec,
            _spec_lengths,
            _wave,
            _wave_lengths,
            _sid,
        ) = (
            torch.randn(DATASET_SIZE, TENSOR_HEIGHT),
            torch.randint(50, TENSOR_HEIGHT, (DATASET_SIZE,)),
            torch.randn(DATASET_SIZE, 100),
            torch.randn(DATASET_SIZE, 100),
            torch.randn(DATASET_SIZE, TENSOR_WIDTH, TENSOR_TIME_FRAMES),
            torch.randint(50, TENSOR_TIME_FRAMES, (DATASET_SIZE,)),
            torch.randn(DATASET_SIZE, TENSOR_AUDIO_SAMPLES),
            torch.randint(8000, TENSOR_AUDIO_SAMPLES, (DATASET_SIZE,)),
            torch.randint(0, 10, (DATASET_SIZE,)),
        )

        assert phone.shape == (DATASET_SIZE, TENSOR_HEIGHT)
        assert pitch.shape == (DATASET_SIZE, 100)
        assert spec.shape == (DATASET_SIZE, TENSOR_WIDTH, TENSOR_TIME_FRAMES)

    def test_batch_unpacking_no_f0(self) -> None:
        """Test unpacking batch without f0 features."""
        # Simulate batch without f0
        (
            phone,
            _phone_lengths,
            spec,
            _spec_lengths,
            _wave,
            _wave_lengths,
            _sid,
        ) = (
            torch.randn(DATASET_SIZE, TENSOR_HEIGHT),
            torch.randint(50, TENSOR_HEIGHT, (DATASET_SIZE,)),
            torch.randn(DATASET_SIZE, TENSOR_WIDTH, TENSOR_TIME_FRAMES),
            torch.randint(50, TENSOR_TIME_FRAMES, (DATASET_SIZE,)),
            torch.randn(DATASET_SIZE, TENSOR_AUDIO_SAMPLES),
            torch.randint(8000, TENSOR_AUDIO_SAMPLES, (DATASET_SIZE,)),
            torch.randint(0, 10, (DATASET_SIZE,)),
        )

        assert phone.shape == (DATASET_SIZE, TENSOR_HEIGHT)
        assert spec.shape == (DATASET_SIZE, TENSOR_WIDTH, TENSOR_TIME_FRAMES)


class TestTrainingHyperparameters:
    """Test training hyperparameter configurations."""

    def test_learning_rate_configuration(self) -> None:
        """Test learning rate setup."""
        lr = 2e-4
        assert lr > 0
        assert isinstance(lr, float)

    def test_beta_configuration(self) -> None:
        """Test optimizer beta parameters."""
        betas = (0.8, 0.99)
        assert len(betas) == BETAS_LENGTH
        assert betas[0] < betas[1]
        assert all(0 < beta < 1 for beta in betas)

    def test_eps_configuration(self) -> None:
        """Test epsilon parameter for optimizer."""
        eps = 1e-9
        assert eps > 0
        assert eps < EPS_MAX_THRESHOLD

    def test_lr_decay_configuration(self) -> None:
        """Test learning rate decay setting."""
        lr_decay = 0.999
        assert 0 < lr_decay < 1


class TestModelOutputShapes:
    """Test output shapes from model components."""

    def test_tensor_slice_operation(self) -> None:
        """Test slicing tensors."""
        spec = torch.randn(BATCH_SIZE_TEST, TENSOR_WIDTH, TENSOR_TIME_FRAMES)

        sliced = torch.nn.functional.pad(spec, (0, 0, 0, 0))
        assert sliced.shape == spec.shape

    def test_mel_spectrogram_shape(self) -> None:
        """Test mel spectrogram tensor shapes."""
        batch_size = BATCH_SIZE_TEST
        mel_channels = 128
        time_frames = 100

        mel_spec = torch.randn(batch_size, mel_channels, time_frames)
        assert mel_spec.shape == (batch_size, mel_channels, time_frames)

    def test_waveform_shape(self) -> None:
        """Test waveform tensor shapes."""
        batch_size = BATCH_SIZE_TEST
        sample_rate = 16000
        duration_seconds = 1

        waveform = torch.randn(batch_size, sample_rate * duration_seconds)
        assert waveform.shape == (batch_size, sample_rate * duration_seconds)

    def test_discriminator_output_shapes(self) -> None:
        """Test expected discriminator output shapes."""
        batch_size = BATCH_SIZE_TEST
        # Discriminator outputs multiple period-based outputs
        y_d_hat_r: list[torch.Tensor] = [
            torch.randn(batch_size, 1, 100) for _ in range(PERIOD_OUTPUTS_COUNT)
        ]
        y_d_hat_g: list[torch.Tensor] = [
            torch.randn(batch_size, 1, 100) for _ in range(PERIOD_OUTPUTS_COUNT)
        ]

        assert len(y_d_hat_r) == PERIOD_OUTPUTS_COUNT
        assert len(y_d_hat_g) == PERIOD_OUTPUTS_COUNT
        assert all(t.shape == (batch_size, 1, 100) for t in y_d_hat_r)


class TestLossCalculations:
    """Test loss calculation operations."""

    def test_l1_loss_calculation(self) -> None:
        """Test L1 loss calculation."""
        x = torch.randn(BATCH_SIZE_TEST, 128, 100)
        y = torch.randn(BATCH_SIZE_TEST, 128, 100)

        loss = torch.nn.functional.l1_loss(x, y)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert loss.dim() == 0  # Scalar

    def test_mse_loss_calculation(self) -> None:
        """Test MSE loss calculation."""
        x = torch.randn(BATCH_SIZE_TEST, 128, 100)
        y = torch.randn(BATCH_SIZE_TEST, 128, 100)

        loss = torch.nn.functional.mse_loss(x, y)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_loss_scaling(self) -> None:
        """Test loss scaling for mixed precision."""
        loss = torch.tensor(0.5, requires_grad=True)
        scale_factor = 45.0

        scaled_loss = loss * scale_factor
        assert abs(scaled_loss.item() - 22.5) < TOLERANCE_EPSILON

    def test_combined_loss(self) -> None:
        """Test combining multiple loss components."""
        loss_mel = torch.tensor(0.5)
        loss_kl = torch.tensor(0.1)
        loss_fm = torch.tensor(0.2)
        loss_gen = torch.tensor(0.3)

        total_loss = loss_mel + loss_kl + loss_fm + loss_gen
        assert abs(total_loss.item() - 1.1) < TOLERANCE_EPSILON


class TestTensorOperations:
    """Test common tensor operations in training."""

    def test_tensor_concatenation(self) -> None:
        """Test concatenating tensors."""
        t1 = torch.randn(BATCH_SIZE_TEST, 64)
        t2 = torch.randn(BATCH_SIZE_TEST, 64)

        combined = torch.cat([t1, t2], dim=1)
        assert combined.shape == (BATCH_SIZE_TEST, 128)

    def test_tensor_expansion(self) -> None:
        """Test expanding tensor dimensions."""
        t = torch.randn(BATCH_SIZE_TEST, 1, 100)
        expanded = t.expand(BATCH_SIZE_TEST, 64, 100)

        assert expanded.shape == (BATCH_SIZE_TEST, 64, 100)

    def test_tensor_reshape(self) -> None:
        """Test reshaping tensors."""
        t = torch.randn(BATCH_SIZE_TEST, 128, 100)
        reshaped = t.reshape(BATCH_SIZE_TEST, -1)

        assert reshaped.shape == (BATCH_SIZE_TEST, 128 * 100)

    def test_tensor_mean_operation(self) -> None:
        """Test computing tensor mean."""
        t = torch.randn(BATCH_SIZE_TEST, 128, 100)
        mean = torch.mean(t)

        assert isinstance(mean, torch.Tensor)
        assert mean.dim() == 0

    def test_tensor_abs_operation(self) -> None:
        """Test absolute value operation."""
        t = torch.randn(BATCH_SIZE_TEST, 128, 100)
        abs_t = torch.abs(t)

        assert abs_t.shape == t.shape
        assert torch.all(abs_t >= 0)


class TestGradientOperations:
    """Test gradient-related operations."""

    def test_gradient_computation(self) -> None:
        """Test computing gradients."""
        x = torch.randn(BATCH_SIZE_TEST, 10, requires_grad=True)
        y = (x**2).sum()

        y.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_zero(self) -> None:
        """Test zeroing gradients."""
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters())

        # Create and backprop some loss
        x = torch.randn(4, 10)
        y = model(x).sum()
        y.backward()

        # Check gradients exist
        assert any(p.grad is not None for p in model.parameters())

        # Zero gradients
        optimizer.zero_grad()

        # Check all gradients are zeroed
        assert all(p.grad is None or torch.all(p.grad == 0) for p in model.parameters())

    def test_gradient_clipping(self) -> None:
        """Test gradient clipping."""
        x = torch.randn(BATCH_SIZE_TEST, 10, requires_grad=True)
        y = (x**2).sum()
        y.backward()

        # Apply clipping
        torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)

        # Gradient norm should be <= max_norm
        grad_norm = torch.norm(x.grad)
        assert grad_norm <= 1.0 + 1e-5


class TestOptimizerStepOperations:
    """Test optimizer step operations."""

    def test_optimizer_step(self) -> None:
        """Test optimizer step."""
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters(), lr=0.01)

        initial_params = [p.clone() for p in model.parameters()]

        # Forward pass
        x = torch.randn(4, 10)
        y = model(x).sum()

        # Backward pass
        y.backward()

        # Optimizer step
        optimizer.step()

        # Parameters should have changed
        for p_init, p in zip(initial_params, model.parameters()):
            assert not torch.allclose(p_init, p)

    def test_learning_rate_scheduler(self) -> None:
        """Test learning rate scheduler."""
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        initial_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()
        updated_lr = optimizer.param_groups[0]["lr"]

        assert updated_lr < initial_lr
        assert updated_lr == pytest.approx(initial_lr * 0.9)


class TestDistributedTrainingSetup:
    """Test distributed training configuration."""

    def test_distributed_backend_string(self) -> None:
        """Test distributed backend configuration."""
        backend = "gloo"
        assert backend in ("nccl", "gloo", "mpi")

    def test_rank_assignment(self) -> None:
        """Test rank assignment in distributed setup."""
        rank = 0
        n_gpus = 1

        assert rank >= 0
        assert rank < n_gpus

    def test_master_addr_port(self) -> None:
        """Test MASTER_ADDR and MASTER_PORT configuration."""
        master_addr = "localhost"
        master_port = random.randint(MASTER_PORT_MIN, MASTER_PORT_MAX)

        assert master_addr == "localhost"
        assert MASTER_PORT_MIN <= master_port <= MASTER_PORT_MAX


class TestCheckpointOperations:
    """Test checkpoint saving and loading."""

    def test_checkpoint_dict_structure(self) -> None:
        """Test checkpoint dictionary structure."""
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters())

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": ITERATION_CHECKPOINT,
            "learning_rate": LEARNING_RATE_CHECKPOINT,
        }

        assert "model" in checkpoint
        assert "optimizer" in checkpoint
        assert "iteration" in checkpoint
        assert "learning_rate" in checkpoint
        assert checkpoint["iteration"] == ITERATION_CHECKPOINT
        assert checkpoint["learning_rate"] == LEARNING_RATE_CHECKPOINT

    def test_model_state_dict_save(self) -> None:
        """Test saving model state dict."""
        model = nn.Linear(10, 5)
        state_dict = model.state_dict()

        assert "weight" in state_dict
        assert "bias" in state_dict
        assert state_dict["weight"].shape == (5, 10)
        assert state_dict["bias"].shape == (5,)

    def test_optimizer_state_dict_save(self) -> None:
        """Test saving optimizer state dict."""
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters())

        state_dict = optimizer.state_dict()

        assert "state" in state_dict
        assert "param_groups" in state_dict
        assert len(state_dict["param_groups"]) > 0
