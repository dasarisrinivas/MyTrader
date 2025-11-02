"""Generate synthetic ES minute data with sentiment for backtesting."""
from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "es_synthetic_with_sentiment.csv"


def trading_minutes_per_day(start: datetime, minutes_per_day: int) -> list[datetime]:
    return [start + timedelta(minutes=i) for i in range(minutes_per_day)]


def main() -> None:
    rng = random.Random(42)
    start_day = datetime(2024, 1, 2, 9, 30)
    trading_minutes = 390  # 6.5 hours
    num_days = 5
    price = 4700.0
    volume = 12_000
    rows = []

    for day in range(num_days):
        day_start = start_day + timedelta(days=day)
        if day_start.weekday() >= 5:
            day_start += timedelta(days=(7 - day_start.weekday()))
        timestamps = trading_minutes_per_day(day_start, trading_minutes)
        trend = rng.uniform(-0.6, 0.8)

        for i, timestamp in enumerate(timestamps):
            intraday_progress = i / trading_minutes
            drift = trend * math.sin(math.pi * (intraday_progress - 0.5))
            shock = rng.uniform(-0.8, 0.8)
            open_price = price
            close_price = max(3800.0, open_price + drift + shock)
            high_price = max(open_price, close_price) + abs(rng.uniform(0, 1.2))
            low_price = min(open_price, close_price) - abs(rng.uniform(0, 1.2))
            volume = max(2_000, volume + rng.randint(-400, 450))

            sentiment_base = math.tanh((close_price - 4700) / 75)
            sentiment_noise = rng.uniform(-0.15, 0.15)
            twitter_sentiment = max(-1.0, min(1.0, sentiment_base + sentiment_noise))
            news_sentiment = max(-1.0, min(1.0, sentiment_base * 0.8 + sentiment_noise * 0.5))

            rows.append(
                (
                    timestamp.isoformat(),
                    round(open_price, 2),
                    round(high_price, 2),
                    round(low_price, 2),
                    round(close_price, 2),
                    int(volume),
                    round(twitter_sentiment, 4),
                    round(news_sentiment, 4),
                )
            )

            price = close_price

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "sentiment_twitter",
                "sentiment_news",
            ]
        )
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
