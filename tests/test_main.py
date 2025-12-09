"""
Tests for the main module.

Tests cover:
- _is_transient_error: Transient error detection (string-based and type-based)
- _execute_with_retry: Retry logic with exponential backoff
"""

import pytest
from unittest.mock import patch, MagicMock, call, Mock
import time
import requests.exceptions

from main import _is_transient_error, _execute_with_retry


class TestIsTransientError:
    """Tests for transient error detection."""

    def test_detects_503_error(self):
        """Test 503 service unavailable is detected."""
        error = Exception("APIError: [503]: The service is currently unavailable.")
        assert _is_transient_error(error) is True

    def test_detects_502_error(self):
        """Test 502 bad gateway is detected."""
        error = Exception("502 Bad Gateway")
        assert _is_transient_error(error) is True

    def test_detects_504_error(self):
        """Test 504 gateway timeout is detected."""
        error = Exception("504 Gateway Timeout")
        assert _is_transient_error(error) is True

    def test_detects_service_unavailable_text(self):
        """Test 'service unavailable' text is detected."""
        error = Exception("The service is temporarily unavailable")
        assert _is_transient_error(error) is True

    def test_string_connection_error_not_transient(self):
        """Test string 'Connection refused' is NOT detected (use type-based instead)."""
        # String-based fallback doesn't include 'connection' patterns
        # Use actual ConnectionError type for type-based detection
        error = Exception("Connection refused")
        assert _is_transient_error(error) is False

    def test_string_timeout_error_not_transient(self):
        """Test string 'timeout' alone is NOT detected (use type-based instead)."""
        # String-based fallback doesn't include 'timeout' alone
        # Use actual TimeoutError type for type-based detection
        error = Exception("Request timeout after 30 seconds")
        assert _is_transient_error(error) is False

    def test_detects_bad_gateway_text(self):
        """Test 'bad gateway' text is detected."""
        error = Exception("bad gateway error from upstream")
        assert _is_transient_error(error) is True

    def test_non_transient_400_error(self):
        """Test 400 bad request is not transient."""
        error = Exception("400 Bad Request: Invalid parameter")
        assert _is_transient_error(error) is False

    def test_non_transient_401_error(self):
        """Test 401 unauthorized is not transient."""
        error = Exception("401 Unauthorized: Invalid API key")
        assert _is_transient_error(error) is False

    def test_non_transient_404_error(self):
        """Test 404 not found is not transient."""
        error = Exception("404 Not Found")
        assert _is_transient_error(error) is False

    def test_non_transient_generic_error(self):
        """Test generic errors are not transient."""
        error = Exception("Something went wrong")
        assert _is_transient_error(error) is False

    def test_non_transient_value_error(self):
        """Test value errors are not transient."""
        error = ValueError("Invalid value provided")
        assert _is_transient_error(error) is False

    def test_case_insensitive(self):
        """Test error detection is case insensitive."""
        error = Exception("SERVICE UNAVAILABLE")
        assert _is_transient_error(error) is True

    # Type-based error detection tests

    def test_detects_builtin_connection_error_type(self):
        """Test built-in ConnectionError is detected by type."""
        error = ConnectionError("Connection failed")
        assert _is_transient_error(error) is True

    def test_detects_connection_refused_error_type(self):
        """Test ConnectionRefusedError is detected by type."""
        error = ConnectionRefusedError("Connection refused")
        assert _is_transient_error(error) is True

    def test_detects_connection_reset_error_type(self):
        """Test ConnectionResetError is detected by type."""
        error = ConnectionResetError("Connection reset by peer")
        assert _is_transient_error(error) is True

    def test_detects_timeout_error_type(self):
        """Test built-in TimeoutError is detected by type."""
        error = TimeoutError("Operation timed out")
        assert _is_transient_error(error) is True

    def test_detects_requests_connection_error_type(self):
        """Test requests.exceptions.ConnectionError is detected by type."""
        error = requests.exceptions.ConnectionError("Failed to connect")
        assert _is_transient_error(error) is True

    def test_detects_requests_timeout_type(self):
        """Test requests.exceptions.Timeout is detected by type."""
        error = requests.exceptions.Timeout("Request timed out")
        assert _is_transient_error(error) is True

    def test_detects_requests_read_timeout_type(self):
        """Test requests.exceptions.ReadTimeout is detected by type."""
        error = requests.exceptions.ReadTimeout("Read timed out")
        assert _is_transient_error(error) is True

    def test_detects_requests_connect_timeout_type(self):
        """Test requests.exceptions.ConnectTimeout is detected by type."""
        error = requests.exceptions.ConnectTimeout("Connect timed out")
        assert _is_transient_error(error) is True

    def test_detects_http_error_502(self):
        """Test requests.exceptions.HTTPError with 502 status is detected."""
        mock_response = Mock()
        mock_response.status_code = 502
        error = requests.exceptions.HTTPError(response=mock_response)
        assert _is_transient_error(error) is True

    def test_detects_http_error_503(self):
        """Test requests.exceptions.HTTPError with 503 status is detected."""
        mock_response = Mock()
        mock_response.status_code = 503
        error = requests.exceptions.HTTPError(response=mock_response)
        assert _is_transient_error(error) is True

    def test_detects_http_error_504(self):
        """Test requests.exceptions.HTTPError with 504 status is detected."""
        mock_response = Mock()
        mock_response.status_code = 504
        error = requests.exceptions.HTTPError(response=mock_response)
        assert _is_transient_error(error) is True

    def test_http_error_400_not_transient(self):
        """Test requests.exceptions.HTTPError with 400 status is not transient."""
        mock_response = Mock()
        mock_response.status_code = 400
        error = requests.exceptions.HTTPError(response=mock_response)
        assert _is_transient_error(error) is False

    def test_http_error_401_not_transient(self):
        """Test requests.exceptions.HTTPError with 401 status is not transient."""
        mock_response = Mock()
        mock_response.status_code = 401
        error = requests.exceptions.HTTPError(response=mock_response)
        assert _is_transient_error(error) is False

    def test_http_error_404_not_transient(self):
        """Test requests.exceptions.HTTPError with 404 status is not transient."""
        mock_response = Mock()
        mock_response.status_code = 404
        error = requests.exceptions.HTTPError(response=mock_response)
        assert _is_transient_error(error) is False

    def test_http_error_500_not_transient(self):
        """Test requests.exceptions.HTTPError with 500 status is not transient (only 502-504)."""
        mock_response = Mock()
        mock_response.status_code = 500
        error = requests.exceptions.HTTPError(response=mock_response)
        assert _is_transient_error(error) is False

    def test_http_error_no_response_not_transient(self):
        """Test HTTPError without response attribute is not transient by type alone."""
        error = requests.exceptions.HTTPError("HTTP error without response")
        # No response attribute with status_code, so falls through to string matching
        # "HTTP error without response" doesn't match transient patterns
        assert _is_transient_error(error) is False


class TestExecuteWithRetry:
    """Tests for the retry execution wrapper."""

    def test_success_on_first_try(self):
        """Test function succeeds on first attempt."""
        mock_func = MagicMock()

        with patch("main.time.sleep"):
            result = _execute_with_retry(mock_func, "test_func")

        assert result is True
        assert mock_func.call_count == 1

    def test_retries_on_transient_error_then_succeeds(self):
        """Test function retries on transient error and eventually succeeds."""
        call_count = [0]

        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("503 Service Unavailable")

        with patch("main.time.sleep"):
            result = _execute_with_retry(flaky_func, "test_func")

        assert result is True
        assert call_count[0] == 2

    def test_retries_multiple_times_then_succeeds(self):
        """Test function retries multiple times before succeeding."""
        call_count = [0]

        def very_flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("502 Bad Gateway")

        with patch("main.time.sleep"):
            result = _execute_with_retry(very_flaky_func, "test_func")

        assert result is True
        assert call_count[0] == 3

    def test_fails_after_max_retries(self):
        """Test function fails after max retries exhausted."""

        def always_fail():
            raise Exception("503 Service Unavailable")

        with patch("main.UPDATE_MAX_RETRIES", 2), patch("main.time.sleep"):
            result = _execute_with_retry(always_fail, "test_func")

        assert result is False

    def test_no_retry_on_non_transient_error(self):
        """Test no retry on non-transient errors."""
        call_count = [0]

        def bad_request():
            call_count[0] += 1
            raise Exception("400 Bad Request")

        with patch("main.time.sleep"):
            result = _execute_with_retry(bad_request, "test_func")

        assert result is False
        assert call_count[0] == 1  # No retries

    def test_exponential_backoff_delays(self):
        """Test that delays follow exponential backoff pattern."""
        sleep_calls = []

        def capture_sleep(duration):
            sleep_calls.append(duration)

        def always_fail():
            raise Exception("503 Service Unavailable")

        with (
            patch("main.UPDATE_MAX_RETRIES", 3),
            patch("main.UPDATE_BASE_DELAY", 1.0),
            patch("main.UPDATE_MAX_DELAY", 100.0),
            patch("main.time.sleep", side_effect=capture_sleep),
            patch("main.random.uniform", return_value=0),
        ):  # No jitter for predictable test
            _execute_with_retry(always_fail, "test_func")

        # Should have 3 sleep calls (for 3 retries)
        assert len(sleep_calls) == 3
        # Delays should be 1, 2, 4 (exponential backoff)
        assert sleep_calls[0] == 1.0
        assert sleep_calls[1] == 2.0
        assert sleep_calls[2] == 4.0

    def test_delay_capped_at_max(self):
        """Test that delay is capped at UPDATE_MAX_DELAY."""
        sleep_calls = []

        def capture_sleep(duration):
            sleep_calls.append(duration)

        def always_fail():
            raise Exception("503 Service Unavailable")

        with (
            patch("main.UPDATE_MAX_RETRIES", 5),
            patch("main.UPDATE_BASE_DELAY", 10.0),
            patch("main.UPDATE_MAX_DELAY", 30.0),
            patch("main.time.sleep", side_effect=capture_sleep),
            patch("main.random.uniform", return_value=0),
        ):
            _execute_with_retry(always_fail, "test_func")

        # With base=10, delays would be 10, 20, 40, 80, 160
        # But capped at 30, so should be 10, 20, 30, 30, 30
        assert sleep_calls[0] == 10.0
        assert sleep_calls[1] == 20.0
        assert sleep_calls[2] == 30.0
        assert sleep_calls[3] == 30.0
        assert sleep_calls[4] == 30.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds some randomness to delays."""
        sleep_calls = []

        def capture_sleep(duration):
            sleep_calls.append(duration)

        def always_fail():
            raise Exception("503 Service Unavailable")

        # Use different random values for each call
        random_values = [0.1, 0.2, 0.15]

        with (
            patch("main.UPDATE_MAX_RETRIES", 3),
            patch("main.UPDATE_BASE_DELAY", 1.0),
            patch("main.UPDATE_MAX_DELAY", 100.0),
            patch("main.time.sleep", side_effect=capture_sleep),
            patch("main.random.uniform", side_effect=random_values),
        ):
            _execute_with_retry(always_fail, "test_func")

        # Delays should include jitter
        # delay 1: 1.0 + 1.0*0.1 = 1.1
        # delay 2: 2.0 + 2.0*0.2 = 2.4
        # delay 3: 4.0 + 4.0*0.15 = 4.6
        assert abs(sleep_calls[0] - 1.1) < 0.01
        assert abs(sleep_calls[1] - 2.4) < 0.01
        assert abs(sleep_calls[2] - 4.6) < 0.01

    def test_logs_warning_on_retry(self):
        """Test that warnings are logged on retry."""
        call_count = [0]

        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("503 Service Unavailable")

        with patch("main.time.sleep"), patch("main.logger") as mock_logger:
            _execute_with_retry(flaky_func, "test_func")

        # Should have logged a warning for the failed attempt
        mock_logger.warning.assert_called_once()
        # Check that the function name was passed as an argument
        call_args = mock_logger.warning.call_args[0]
        # The format string is first, then the arguments
        assert call_args[1] == "test_func"  # func_name argument
        assert call_args[2] == 1  # attempt number

    def test_logs_error_on_final_failure(self):
        """Test that error is logged when all retries fail."""

        def always_fail():
            raise Exception("503 Service Unavailable")

        with (
            patch("main.UPDATE_MAX_RETRIES", 1),
            patch("main.time.sleep"),
            patch("main.logger") as mock_logger,
        ):
            _execute_with_retry(always_fail, "test_func")

        # Should have logged an error
        mock_logger.error.assert_called()

    def test_different_transient_errors(self):
        """Test retry works with different types of transient errors."""
        # Use actual exception types for type-based detection
        errors = [
            Exception("502 Bad Gateway"),  # String-based (502)
            ConnectionError("Connection failed"),  # Type-based
            TimeoutError("Operation timed out"),  # Type-based
        ]
        call_count = [0]

        def rotating_errors():
            call_count[0] += 1
            if call_count[0] <= len(errors):
                raise errors[call_count[0] - 1]

        with patch("main.time.sleep"):
            result = _execute_with_retry(rotating_errors, "test_func")

        assert result is True
        assert call_count[0] == len(errors) + 1  # All errors + final success
