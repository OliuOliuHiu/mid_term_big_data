import vaex

def trips_by_zone(df):
    return df.groupby(
        "pickup_zone",
        agg={"trips": vaex.agg.count()}
    ).to_pandas_df()

def revenue_by_zone(df):
    return df.groupby(
        "pickup_zone",
        agg={"revenue": vaex.agg.sum("fare")}
    ).to_pandas_df()

def trips_by_hour(df):
    return df.groupby(
        "hour",
        agg={"trips": vaex.agg.count()}
    ).to_pandas_df().sort_values("hour")

def avg_fare_by_vehicle(df):
    return df.groupby(
        "vehicle_type",
        agg={"avg_fare": vaex.agg.mean("fare")}
    ).to_pandas_df()

def surge_vs_fare(df):
    return df.groupby(
        "surge_multiplier",
        agg={"avg_fare": vaex.agg.mean("fare")}
    ).to_pandas_df()

def weather_impact(df):
    return df.groupby(
        "weather_condition",
        agg={
            "trips": vaex.agg.count(),
            "avg_fare": vaex.agg.mean("fare")
        }
    ).to_pandas_df()
