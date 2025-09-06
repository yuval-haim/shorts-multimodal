import pandas as pd

def to_week(dt_series: pd.Series) -> pd.Series:
    # ISO week start Monday; convert to Monday of that week
    s = pd.to_datetime(dt_series, utc=True)
    monday = s.dt.tz_convert('UTC').dt.to_period('W-MON').dt.start_time
    return monday.dt.tz_localize('UTC')
