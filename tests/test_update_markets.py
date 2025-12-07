"""
Tests for the update_markets module.

Tests cover:
- update_sheet: DataFrame padding and sheet updates
- fetch_and_process_data: End-to-end market data processing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import sys


class TestUpdateSheet:
    """Tests for the update_sheet function."""

    def test_padding_when_existing_sheet_is_larger(self):
        """Test that data is padded when existing sheet has more rows/cols."""
        with patch.dict(sys.modules, {
            'data_updater': MagicMock(),
            'data_updater.trading_utils': MagicMock(),
            'data_updater.google_utils': MagicMock(),
            'data_updater.find_markets': MagicMock(),
            'gspread_dataframe': MagicMock(),
        }):
            mock_set_with_dataframe = MagicMock()
            with patch('gspread_dataframe.set_with_dataframe', mock_set_with_dataframe):
                from update_markets import update_sheet

                mock_worksheet = Mock()
                mock_worksheet.get_all_values.return_value = [
                    ['a', 'b', 'c', 'd'],
                    ['1', '2', '3', '4'],
                    ['5', '6', '7', '8'],
                    ['9', '10', '11', '12'],
                ]

                data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
                update_sheet(data, mock_worksheet)

                mock_set_with_dataframe.assert_called_once()
                call_args = mock_set_with_dataframe.call_args
                padded_df = call_args[0][1]
                assert padded_df.shape == (4, 4)

    def test_padding_when_existing_sheet_is_smaller(self):
        """Test that data is padded when new data has more rows/cols."""
        with patch.dict(sys.modules, {
            'data_updater': MagicMock(),
            'data_updater.trading_utils': MagicMock(),
            'data_updater.google_utils': MagicMock(),
            'data_updater.find_markets': MagicMock(),
            'gspread_dataframe': MagicMock(),
        }):
            mock_set_with_dataframe = MagicMock()
            with patch('gspread_dataframe.set_with_dataframe', mock_set_with_dataframe):
                from update_markets import update_sheet

                mock_worksheet = Mock()
                mock_worksheet.get_all_values.return_value = [
                    ['a', 'b'],
                    ['1', '2'],
                ]

                data = pd.DataFrame({
                    'col1': [1, 2, 3, 4],
                    'col2': [5, 6, 7, 8],
                    'col3': [9, 10, 11, 12],
                    'col4': [13, 14, 15, 16],
                })
                update_sheet(data, mock_worksheet)

                mock_set_with_dataframe.assert_called_once()
                call_args = mock_set_with_dataframe.call_args
                padded_df = call_args[0][1]
                assert padded_df.shape == (4, 4)

    def test_with_empty_existing_data(self):
        """Test handling when existing sheet has no data."""
        with patch.dict(sys.modules, {
            'data_updater': MagicMock(),
            'data_updater.trading_utils': MagicMock(),
            'data_updater.google_utils': MagicMock(),
            'data_updater.find_markets': MagicMock(),
            'gspread_dataframe': MagicMock(),
        }):
            mock_set_with_dataframe = MagicMock()
            with patch('gspread_dataframe.set_with_dataframe', mock_set_with_dataframe):
                from update_markets import update_sheet

                mock_worksheet = Mock()
                mock_worksheet.get_all_values.return_value = []

                data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
                update_sheet(data, mock_worksheet)

                mock_set_with_dataframe.assert_called_once()
                call_args = mock_set_with_dataframe.call_args
                assert call_args[1]['resize'] is True


class TestFetchAndProcessData:
    """Tests for the fetch_and_process_data function."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up common mocks for all tests in this class."""
        self.mock_spreadsheet = MagicMock()
        self.mock_client = MagicMock()
        self.mock_worksheet = MagicMock()
        self.mock_spreadsheet.worksheet.return_value = self.mock_worksheet

        self.mock_modules = {
            'data_updater': MagicMock(),
            'data_updater.trading_utils': MagicMock(),
            'data_updater.google_utils': MagicMock(),
            'data_updater.find_markets': MagicMock(),
            'gspread_dataframe': MagicMock(),
        }

    def _create_sample_market_df(self, num_rows=100):
        """Create a sample DataFrame with the required columns."""
        return pd.DataFrame({
            'question': [f'Question {i}' for i in range(num_rows)],
            'answer1': ['Yes'] * num_rows,
            'answer2': ['No'] * num_rows,
            'spread': [0.02] * num_rows,
            'rewards_daily_rate': [0.01] * num_rows,
            'gm_reward_per_100': [10.0] * num_rows,
            'sm_reward_per_100': [5.0] * num_rows,
            'bid_reward_per_100': [3.0] * num_rows,
            'ask_reward_per_100': [3.0] * num_rows,
            'volatility_sum': [15.0] * num_rows,
            'min_size': [5.0] * num_rows,
            '1_hour': [0.5] * num_rows,
            '3_hour': [1.0] * num_rows,
            '6_hour': [1.5] * num_rows,
            '12_hour': [2.0] * num_rows,
            '24_hour': [5.0] * num_rows,
            '7_day': [5.0] * num_rows,
            '14_day': [5.0] * num_rows,
            '30_day': [10.0] * num_rows,
            'best_bid': [0.45] * num_rows,
            'best_ask': [0.55] * num_rows,
            'volatility_price': [0.5] * num_rows,
            'max_spread': [5.0] * num_rows,
            'tick_size': [0.01] * num_rows,
            'neg_risk': ['FALSE'] * num_rows,
            'market_slug': [f'market-{i}' for i in range(num_rows)],
            'token1': [f'token1_{i}' for i in range(num_rows)],
            'token2': [f'token2_{i}' for i in range(num_rows)],
            'condition_id': [f'condition_{i}' for i in range(num_rows)],
        })

    def test_updates_sheets_when_enough_markets(self):
        """Test that sheets are updated when there are more than 50 markets."""
        with patch.dict(sys.modules, self.mock_modules):
            sample_df = self._create_sample_market_df(100)

            with patch('update_markets.get_spreadsheet', return_value=self.mock_spreadsheet), \
                 patch('update_markets.get_clob_client', return_value=self.mock_client), \
                 patch('update_markets.get_sel_df', return_value=pd.DataFrame()), \
                 patch('update_markets.get_all_markets', return_value=pd.DataFrame()), \
                 patch('update_markets.get_all_results', return_value=[]), \
                 patch('update_markets.get_markets', return_value=(sample_df, sample_df)), \
                 patch('update_markets.add_volatility_to_df', return_value=sample_df), \
                 patch('update_markets.update_sheet') as mock_update_sheet:

                from update_markets import fetch_and_process_data
                fetch_and_process_data()

                assert mock_update_sheet.call_count == 3

    def test_skips_update_when_few_markets(self):
        """Test that sheets are NOT updated when there are fewer than 50 markets."""
        with patch.dict(sys.modules, self.mock_modules):
            sample_df = self._create_sample_market_df(30)

            with patch('update_markets.get_spreadsheet', return_value=self.mock_spreadsheet), \
                 patch('update_markets.get_clob_client', return_value=self.mock_client), \
                 patch('update_markets.get_sel_df', return_value=pd.DataFrame()), \
                 patch('update_markets.get_all_markets', return_value=pd.DataFrame()), \
                 patch('update_markets.get_all_results', return_value=[]), \
                 patch('update_markets.get_markets', return_value=(sample_df, sample_df)), \
                 patch('update_markets.add_volatility_to_df', return_value=sample_df), \
                 patch('update_markets.update_sheet') as mock_update_sheet:

                from update_markets import fetch_and_process_data
                fetch_and_process_data()

                mock_update_sheet.assert_not_called()

    def test_filters_high_volatility_markets(self):
        """Test that markets with volatility_sum >= 20 are filtered for volatility sheet."""
        with patch.dict(sys.modules, self.mock_modules):
            sample_df = self._create_sample_market_df(100)
            sample_df.loc[0:49, '24_hour'] = 10.0
            sample_df.loc[0:49, '7_day'] = 10.0
            sample_df.loc[0:49, '14_day'] = 10.0
            sample_df.loc[50:99, '24_hour'] = 3.0
            sample_df.loc[50:99, '7_day'] = 3.0
            sample_df.loc[50:99, '14_day'] = 3.0

            with patch('update_markets.get_spreadsheet', return_value=self.mock_spreadsheet), \
                 patch('update_markets.get_clob_client', return_value=self.mock_client), \
                 patch('update_markets.get_sel_df', return_value=pd.DataFrame()), \
                 patch('update_markets.get_all_markets', return_value=pd.DataFrame()), \
                 patch('update_markets.get_all_results', return_value=[]), \
                 patch('update_markets.get_markets', return_value=(sample_df, sample_df)), \
                 patch('update_markets.add_volatility_to_df', return_value=sample_df), \
                 patch('update_markets.update_sheet') as mock_update_sheet:

                from update_markets import fetch_and_process_data
                fetch_and_process_data()

                call_args_list = mock_update_sheet.call_args_list
                volatility_df = call_args_list[1][0][0]
                assert len(volatility_df) == 50
                assert all(volatility_df['volatility_sum'] < 20)

    def test_sorts_by_gm_reward(self):
        """Test that final dataframes are sorted by gm_reward_per_100 descending."""
        with patch.dict(sys.modules, self.mock_modules):
            sample_df = self._create_sample_market_df(100)
            sample_df['gm_reward_per_100'] = list(range(100))

            with patch('update_markets.get_spreadsheet', return_value=self.mock_spreadsheet), \
                 patch('update_markets.get_clob_client', return_value=self.mock_client), \
                 patch('update_markets.get_sel_df', return_value=pd.DataFrame()), \
                 patch('update_markets.get_all_markets', return_value=pd.DataFrame()), \
                 patch('update_markets.get_all_results', return_value=[]), \
                 patch('update_markets.get_markets', return_value=(sample_df, sample_df)), \
                 patch('update_markets.add_volatility_to_df', return_value=sample_df), \
                 patch('update_markets.update_sheet') as mock_update_sheet:

                from update_markets import fetch_and_process_data
                fetch_and_process_data()

                call_args_list = mock_update_sheet.call_args_list
                all_markets_df = call_args_list[0][0][0]
                assert all_markets_df['gm_reward_per_100'].iloc[0] == 99
                assert all_markets_df['gm_reward_per_100'].iloc[-1] == 0
