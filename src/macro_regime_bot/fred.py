"""Small, typed client for the FRED observations endpoint."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests


class FredError(RuntimeError):
    """Raised when FRED cannot return usable observations."""


class FredClient:
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        timeout: float = 30,
    ) -> None:
        if not api_key:
            raise ValueError("A FRED API key is required")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout = timeout

    def fetch_series(self, series_id: str, start_date: str = "2000-01-01") -> pd.Series:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "sort_order": "asc",
        }
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise FredError(f"Unable to fetch {series_id} from FRED: {exc}") from exc

        if "error_message" in payload:
            raise FredError(f"FRED error for {series_id}: {payload['error_message']}")

        observations = payload.get("observations")
        if not isinstance(observations, list) or not observations:
            raise FredError(f"FRED returned no observations for {series_id}")

        dated_values = [
            (item.get("date"), item.get("value"))
            for item in observations
            if item.get("value") not in (None, ".")
        ]
        if not dated_values:
            raise FredError(f"FRED returned only missing observations for {series_id}")

        dates = pd.to_datetime([date for date, _ in dated_values])
        values = pd.to_numeric(pd.Series([value for _, value in dated_values]), errors="coerce")
        result = pd.Series(values.to_numpy(), index=dates, name=series_id, dtype=float).dropna()
        if result.empty:
            raise FredError(f"FRED returned no numeric observations for {series_id}")
        return result

    def fetch_many(
        self, series_ids: list[str], start_date: str = "2000-01-01"
    ) -> dict[str, pd.Series]:
        return {
            series_id: self.fetch_series(series_id, start_date=start_date)
            for series_id in series_ids
        }
