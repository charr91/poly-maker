"""
Tests for the Google Sheets utilities module.

Tests cover:
- get_spreadsheet: Authentication and read-only mode
- ReadOnlySpreadsheet: URL parsing and worksheet access
- ReadOnlyWorksheet: CSV fetching with fallback logic
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from poly_utils.google_utils import (
    get_spreadsheet,
    ReadOnlySpreadsheet,
    ReadOnlyWorksheet,
)


class TestGetSpreadsheet:
    """Tests for the get_spreadsheet function."""

    @patch("poly_utils.google_utils.gspread")
    @patch("poly_utils.google_utils.Credentials")
    @patch("poly_utils.google_utils.os.path.exists")
    @patch("poly_utils.google_utils.os.getenv")
    def test_successful_authentication(self, mock_getenv, mock_exists, mock_creds, mock_gspread):
        """Test successful spreadsheet authentication with credentials."""
        mock_getenv.return_value = "https://docs.google.com/spreadsheets/d/abc123"
        mock_exists.return_value = True
        mock_creds_instance = Mock()
        mock_creds.from_service_account_file.return_value = mock_creds_instance
        mock_client = Mock()
        mock_gspread.authorize.return_value = mock_client
        mock_spreadsheet = Mock()
        mock_client.open_by_url.return_value = mock_spreadsheet

        result = get_spreadsheet()

        assert result == mock_spreadsheet
        mock_creds.from_service_account_file.assert_called_once()
        mock_gspread.authorize.assert_called_once_with(mock_creds_instance)
        mock_client.open_by_url.assert_called_once()

    @patch("poly_utils.google_utils.os.path.exists")
    @patch("poly_utils.google_utils.os.getenv")
    def test_read_only_mode_when_no_credentials(self, mock_getenv, mock_exists):
        """Test that read_only mode returns ReadOnlySpreadsheet when credentials missing."""
        mock_getenv.return_value = "https://docs.google.com/spreadsheets/d/abc123"
        mock_exists.return_value = False

        result = get_spreadsheet(read_only=True)

        assert isinstance(result, ReadOnlySpreadsheet)

    @patch("poly_utils.google_utils.os.getenv")
    def test_raises_value_error_when_no_url(self, mock_getenv):
        """Test that ValueError is raised when SPREADSHEET_URL not set."""
        mock_getenv.return_value = None

        with pytest.raises(ValueError, match="SPREADSHEET_URL environment variable is not set"):
            get_spreadsheet()

    @patch("poly_utils.google_utils.os.path.exists")
    @patch("poly_utils.google_utils.os.getenv")
    def test_raises_file_not_found_when_no_credentials_and_not_read_only(
        self, mock_getenv, mock_exists
    ):
        """Test that FileNotFoundError is raised when credentials missing and not read_only."""
        mock_getenv.return_value = "https://docs.google.com/spreadsheets/d/abc123"
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError):
            get_spreadsheet(read_only=False)


class TestReadOnlySpreadsheet:
    """Tests for the ReadOnlySpreadsheet class."""

    def test_extracts_sheet_id_from_valid_url(self):
        """Test that sheet ID is correctly extracted from valid URL."""
        url = "https://docs.google.com/spreadsheets/d/1Kt6yGY7CZpB75cLJJAdWo7LSp9Oz7pjqfuVWwgtn7Ns/edit"
        spreadsheet = ReadOnlySpreadsheet(url)

        assert spreadsheet.sheet_id == "1Kt6yGY7CZpB75cLJJAdWo7LSp9Oz7pjqfuVWwgtn7Ns"

    def test_raises_value_error_for_invalid_url(self):
        """Test that ValueError is raised for invalid URL format."""
        with pytest.raises(ValueError, match="Invalid Google Sheets URL"):
            ReadOnlySpreadsheet("https://invalid-url.com")

    def test_worksheet_returns_readonly_worksheet(self):
        """Test that worksheet() returns a ReadOnlyWorksheet instance."""
        url = "https://docs.google.com/spreadsheets/d/abc123/edit"
        spreadsheet = ReadOnlySpreadsheet(url)
        worksheet = spreadsheet.worksheet("Selected Markets")

        assert isinstance(worksheet, ReadOnlyWorksheet)
        assert worksheet.title == "Selected Markets"
        assert worksheet.sheet_id == "abc123"


class TestReadOnlyWorksheetGetAllRecords:
    """Tests for ReadOnlyWorksheet.get_all_records()."""

    @patch("poly_utils.google_utils.time.sleep")
    @patch("poly_utils.google_utils.requests.get")
    def test_successful_fetch_first_url(self, mock_get, mock_sleep):
        """Test successful data fetch from first URL attempt."""
        mock_response = Mock()
        mock_response.text = "col1,col2\nval1,val2\nval3,val4"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        worksheet = ReadOnlyWorksheet("abc123", "All Markets")
        result = worksheet.get_all_records()

        assert len(result) == 2
        assert result[0] == {"col1": "val1", "col2": "val2"}
        assert result[1] == {"col1": "val3", "col2": "val4"}
        mock_sleep.assert_not_called()

    @patch("poly_utils.google_utils.time.sleep")
    @patch("poly_utils.google_utils.requests.get")
    def test_fallback_to_second_url(self, mock_get, mock_sleep):
        """Test fallback to subsequent URL when first fails."""
        mock_success_response = Mock()
        mock_success_response.text = "col1,col2\nval1,val2"
        mock_success_response.raise_for_status = Mock()

        mock_get.side_effect = [Exception("First URL failed"), mock_success_response]

        worksheet = ReadOnlyWorksheet("abc123", "All Markets")
        result = worksheet.get_all_records()

        assert len(result) == 1
        assert result[0] == {"col1": "val1", "col2": "val2"}
        mock_sleep.assert_called_once()

    @patch("poly_utils.google_utils.time.sleep")
    @patch("poly_utils.google_utils.requests.get")
    def test_hyperparameters_column_validation(self, mock_get, mock_sleep):
        """Test that Hyperparameters sheet validates expected columns."""
        mock_response = Mock()
        mock_response.text = "type,param,value\nparam_type,spread,0.5"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        worksheet = ReadOnlyWorksheet("abc123", "Hyperparameters")
        result = worksheet.get_all_records()

        assert len(result) == 1
        assert result[0] == {"type": "param_type", "param": "spread", "value": 0.5}

    @patch("poly_utils.google_utils.time.sleep")
    @patch("poly_utils.google_utils.requests.get")
    def test_hyperparameters_rejects_wrong_columns(self, mock_get, mock_sleep):
        """Test that Hyperparameters sheet rejects data with wrong columns."""
        mock_wrong_response = Mock()
        mock_wrong_response.text = "wrong_col1,wrong_col2\nval1,val2"
        mock_wrong_response.raise_for_status = Mock()

        mock_correct_response = Mock()
        mock_correct_response.text = "type,param,value\nparam_type,spread,0.5"
        mock_correct_response.raise_for_status = Mock()

        mock_get.side_effect = [mock_wrong_response, mock_correct_response]

        worksheet = ReadOnlyWorksheet("abc123", "Hyperparameters")
        result = worksheet.get_all_records()

        assert len(result) == 1
        assert result[0]["param"] == "spread"

    @patch("poly_utils.google_utils.time.sleep")
    @patch("poly_utils.google_utils.requests.get")
    def test_returns_empty_list_on_all_failures(self, mock_get, mock_sleep):
        """Test that empty list is returned when all URL attempts fail."""
        mock_get.side_effect = Exception("All URLs failed")

        worksheet = ReadOnlyWorksheet("abc123", "All Markets")
        result = worksheet.get_all_records()

        assert result == []


class TestReadOnlyWorksheetGetAllValues:
    """Tests for ReadOnlyWorksheet.get_all_values()."""

    @patch("poly_utils.google_utils.requests.get")
    def test_successful_fetch_returns_headers_and_data(self, mock_get):
        """Test successful fetch returns headers as first row plus data rows."""
        mock_response = Mock()
        mock_response.text = "col1,col2\nval1,val2\nval3,val4"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        worksheet = ReadOnlyWorksheet("abc123", "All Markets")
        result = worksheet.get_all_values()

        assert len(result) == 3
        assert result[0] == ["col1", "col2"]
        assert result[1] == ["val1", "val2"]
        assert result[2] == ["val3", "val4"]

    @patch("poly_utils.google_utils.requests.get")
    def test_returns_empty_list_on_exception(self, mock_get):
        """Test that empty list is returned on exception."""
        mock_get.side_effect = Exception("Network error")

        worksheet = ReadOnlyWorksheet("abc123", "All Markets")
        result = worksheet.get_all_values()

        assert result == []
