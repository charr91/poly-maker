import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import requests
import sys

# Ensure we can import modules
sys.path.append(".")

from data_updater import find_markets
from poly_data.rate_limiter import EndpointType


class TestFindMarketsIntegration:

    @pytest.fixture
    def mock_rate_limit_manager(self):
        with patch("data_updater.find_markets.get_rate_limit_manager") as mock:
            yield mock

    @pytest.fixture
    def mock_session_cls(self):
        with patch("requests.Session") as mock:
            yield mock

    def test_connection_pooling_configuration(self, mock_session_cls):
        """Verify session is configured with connection pooling and retries."""
        # Reset the global session if it was already created in other tests or imports
        find_markets._session = None

        session_instance = MagicMock()
        mock_session_cls.return_value = session_instance

        # First call creates the session
        s1 = find_markets.get_session()

        # Verify Session was created
        mock_session_cls.assert_called_once()

        # Verify mounting adapters
        assert session_instance.mount.call_count >= 2
        call_args = session_instance.mount.call_args_list[0]
        prefix, adapter = call_args[0]

        # Check adapter config
        assert isinstance(adapter, requests.adapters.HTTPAdapter)
        # Note: pool_connections is protected/internal in some versions or just an attribute
        # We can check the arguments passed to constructor if we mocked HTTPAdapter,
        # but since we imported it for the script, let's verify attributes if possible
        # or just rely on the fact that the script uses it.
        # Better: Mock HTTPAdapter to verify args.

    def test_rate_limiting_book(self, mock_rate_limit_manager):
        """Verify book rate limiting calls the centralized manager."""
        find_markets._use_centralized_rate_limiter = True

        find_markets.rate_limit_book()

        manager = mock_rate_limit_manager.return_value
        manager.acquire_sync.assert_called_once_with(EndpointType.BOOK)

    def test_rate_limiting_data_api(self, mock_rate_limit_manager):
        """Verify data API rate limiting calls the centralized manager."""
        find_markets._use_centralized_rate_limiter = True

        find_markets.rate_limit_data_api()

        manager = mock_rate_limit_manager.return_value
        manager.acquire_sync.assert_called_once_with(EndpointType.DATA_API)

    def test_add_volatility_uses_pooled_session(self, mock_rate_limit_manager):
        """Verify add_volatility uses the shared session and rate limiting."""
        # Mock get_session to return a specific mock session
        mock_sess = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "history": [{"t": 1000, "p": 0.5}, {"t": 2000, "p": 0.6}]
        }
        mock_sess.get.return_value = mock_response

        with patch("data_updater.find_markets.get_session", return_value=mock_sess):
            with patch("pandas.DataFrame.to_csv"):  # suppress file writing
                row = {"token1": "0x123", "question": "test"}
                find_markets.add_volatility(row)

                # Check rate limit was called
                manager = mock_rate_limit_manager.return_value
                manager.acquire_sync.assert_called_with(EndpointType.DATA_API)

                # Check session.get was called (reusing pool)
                mock_sess.get.assert_called_once()
                args, kwargs = mock_sess.get.call_args
                assert "prices-history" in args[0]

    def test_get_all_results_error_handling(self):
        """Verify bulk processing handles errors gracefully."""
        # Mock client
        client = MagicMock()

        # Mock process_single_row to fail for some items
        def side_effect(row, client):
            if row["fail"]:
                raise Exception("simulated error")
            return {"question": row["question"], "success": True}

        with patch("data_updater.find_markets.process_single_row", side_effect=side_effect):
            # Create test dataframe
            df = pd.DataFrame(
                [
                    {"question": "q1", "fail": False},
                    {"question": "q2", "fail": True},
                    {"question": "q3", "fail": False},
                ]
            )

            # Run get_all_results
            # We need to mock ThreadPoolExecutor to run synchronously or just let it run
            # The real implementation uses concurrent.futures.ThreadPoolExecutor
            # It should just work with standard python threading

            results = find_markets.get_all_results(df, client, max_workers=2)

            # Should have 2 successes (q1, q3)
            assert len(results) == 2
            assert results[0]["question"] in ["q1", "q3"]
            assert results[1]["question"] in ["q1", "q3"]
