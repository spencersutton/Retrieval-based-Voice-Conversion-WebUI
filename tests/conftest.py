"""
Pytest configuration and fixtures for train tests.
"""

import sys
from unittest.mock import MagicMock, patch

# Mock the command line arguments that train.py expects
sys.argv = ["pytest"]

# Mock argparse to avoid requiring command line arguments
mock_parser = MagicMock()
mock_args = MagicMock()
mock_args.gpus = "0"
mock_args.model_dir = "/tmp/test_model"
mock_args.if_f0 = 1
mock_args.if_cache_data_in_gpu = False

with patch("argparse.ArgumentParser") as mock_argparse:
    mock_argparse.return_value.parse_args.return_value = mock_args
