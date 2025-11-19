from numpy import cos, pi, sin


def add_cyclical_features(df, time_column: str) -> None:
    df["hour_sin"] = sin(2 * pi * df[time_column] / (60 * 60 * 24))
    df["hour_cos"] = cos(2 * pi * df[time_column] / (60 * 60 * 24))
