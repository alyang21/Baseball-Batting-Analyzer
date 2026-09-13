import pandas as pd

# Load and clean data: the source file duplicates every player's stats
# across 2024 and 2025 with identical values, so we treat it as one
# season snapshot rather than two real years.
_raw = pd.read_csv("data/mlb_bat_tracking_2024_2025.csv")
df = _raw[_raw["season"] == 2024].drop(columns=["season"]).reset_index(drop=True)


def _find_player(name: str) -> pd.DataFrame:
    """Find a player by name. Prefers exact match, then startswith,
    then partial match, to avoid 'Cruz' accidentally matching 'De La Cruz'."""
    exact = df[df["name"].str.lower() == name.lower()]
    if not exact.empty:
        return exact
    starts = df[df["name"].str.lower().str.startswith(name.lower())]
    if not starts.empty:
        return starts
    return df[df["name"].str.contains(name, case=False, na=False)]


def get_player_stats(name: str) -> dict:
    """Return all stats for a single player."""
    matches = _find_player(name)
    if matches.empty:
        return {"error": f"No player found matching '{name}'"}
    if matches["name"].nunique() > 1:
        return {"error": f"'{name}' is ambiguous, matches: {list(matches['name'].unique())}. Be more specific."}
    return matches.to_dict(orient="records")[0]


def compare_players(name1: str, name2: str, metric: str = None) -> dict:
    """Compare two players, either on one metric or all metrics."""
    p1 = _find_player(name1)
    p2 = _find_player(name2)
    if p1.empty or p2.empty:
        return {"error": "One or both players not found"}
    if p1["name"].nunique() > 1:
        return {"error": f"'{name1}' is ambiguous, matches: {list(p1['name'].unique())}"}
    if p2["name"].nunique() > 1:
        return {"error": f"'{name2}' is ambiguous, matches: {list(p2['name'].unique())}"}
    if metric:
        return {name1: float(p1[metric].iloc[0]), name2: float(p2[metric].iloc[0])}


def get_leaderboard(metric: str, top_n: int = 10) -> dict:
    if metric not in df.columns:
        return {"error": f"'{metric}' is not a valid column. Valid columns are: {list(df.columns)}"}
    top = df.nlargest(top_n, metric)[["name", metric]]
    return top.to_dict(orient="records")


def get_correlation(metric1: str, metric2: str) -> dict:
    for m in (metric1, metric2):
        if m not in df.columns:
            return {"error": f"'{m}' is not a valid column. Valid columns are: {list(df.columns)}"}
    corr = df[metric1].corr(df[metric2])
    return {"metric1": metric1, "metric2": metric2, "correlation": round(float(corr), 3)}


def get_summary_stats(metric: str) -> dict:
    if metric not in df.columns:
        return {"error": f"'{metric}' is not a valid column. Valid columns are: {list(df.columns)}"}
    return {k: float(v) for k, v in df[metric].describe().to_dict().items()}


if __name__ == "__main__":
    print("--- leaderboard: top 5 bat speed ---")
    print(get_leaderboard("avg_bat_speed", top_n=5))

    print("\n--- correlation: bat speed vs run value ---")
    print(get_correlation("avg_bat_speed", "batter_run_value"))

    print("\n--- single player ---")
    print(get_player_stats("Schwarber"))

    print("\n--- ambiguous name, should return error ---")
    print(get_player_stats("Cruz"))

    print("\n--- unambiguous compare ---")
    print(compare_players("Schwarber", "Walker, Jordan", metric="avg_bat_speed"))

    print("\n--- summary stats ---")
    print(get_summary_stats("avg_bat_speed"))