from unittest.mock import Mock

import pytest

from macro_regime_bot.fred import FredClient, FredError


def response(payload):
    result = Mock()
    result.raise_for_status.return_value = None
    result.json.return_value = payload
    return result


def test_fred_parses_dates_numbers_and_missing_values() -> None:
    session = Mock()
    session.get.return_value = response(
        {
            "observations": [
                {"date": "2025-01-01", "value": "1.25"},
                {"date": "2025-01-02", "value": "."},
                {"date": "2025-01-03", "value": "2.5"},
            ]
        }
    )
    series = FredClient("test-key", session=session).fetch_series("TEST")
    assert series.tolist() == [1.25, 2.5]
    assert series.index[0].date().isoformat() == "2025-01-01"
    params = session.get.call_args.kwargs["params"]
    assert params["series_id"] == "TEST"
    assert params["file_type"] == "json"


def test_fred_api_error_is_descriptive() -> None:
    session = Mock()
    session.get.return_value = response({"error_message": "Bad API key"})
    with pytest.raises(FredError, match="Bad API key"):
        FredClient("bad-key", session=session).fetch_series("CPIAUCSL")
