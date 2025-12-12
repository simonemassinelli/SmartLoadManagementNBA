import pandas as pd
import numpy as np
from pandas import read_csv


def load_data():
    data_path = "../../data/initial_FE.csv"
    return read_csv(data_path)


def summarize_numeric_feature(series, bins=10):
    series = pd.to_numeric(series, errors="coerce")
    stats = series.describe()
    print(f"Count: {int(stats['count']):,}")
    print(f"Min: {stats['min']:.3f}")
    print(f"25%: {stats['25%']:.3f}")
    print(f"50% (median): {stats['50%']:.3f}")
    print(f"75%: {stats['75%']:.3f}")
    print(f"Max: {stats['max']:.3f}")
    print(f"Mean: {stats['mean']:.3f}")
    print(f"Std: {stats['std']:.3f}")

    print("\nHistogram distribution:")
    hist, bin_edges = np.histogram(series.dropna(), bins=bins)
    total = len(series)
    for count, edge_start, edge_end in zip(hist, bin_edges[:-1], bin_edges[1:]):
        pct = count / total * 100
        print(f"{edge_start:7.2f} – {edge_end:7.2f}: {count:7,} ({pct:4.1f}%)")


def investigate_features(df, features):
    for feature, specs in features.items():
        if feature not in df.columns:
            print(f"{feature} : missing")
            print()
            continue

        print(feature)

        if specs["type"] == "binary":
            vals = pd.to_numeric(df[feature], errors="coerce")
            dist = vals.value_counts(dropna=False)
            total = len(df)
            print("Distribution:")
            for val in specs["expected_values"]:
                count = dist.get(val, 0) + dist.get(float(val), 0)
                pct = count / total * 100
                print(f"{val}: {int(count):,} ({pct:.1f}%)")

        elif specs["type"] == "numeric":
            summarize_numeric_feature(df[feature])

        missing = df[feature].isna().sum()
        if missing > 0:
            print(f"Missing: {missing:,}")
        print()



def check_basic_features(df):
    features = {
        "TEAM_WON": {"type": "binary", "expected_values": [0, 1]},
        "TOTAL_POINTS": {"type": "numeric"},
        "HIGH_SCORING": {"type": "binary", "expected_values": [0, 1]},
        "BLOWOUT": {"type": "binary", "expected_values": [0, 1]},
        "CLOSE_GAME": {"type": "binary", "expected_values": [0, 1]},
        "LOW_USAGE": {"type": "binary", "expected_values": [0, 1]},
    }
    investigate_features(df, features)


def check_players_features(df):
    features = {
        "YEARS_IN_LEAGUE": {"type": "numeric"},
        "IS_ROOKIE": {"type": "binary", "expected_values": [0, 1]},
        "IS_VETERAN": {"type": "binary", "expected_values": [0, 1]},
        "AVAILABILITY": {"type": "numeric"},
    }
    investigate_features(df, features)


def check_time_features(df):
    features = {
        "DAYS_REST": {"type": "numeric"},
        "BACK_TO_BACK": {"type": "binary", "expected_values": [0, 1]},
        "WELL_RESTED": {"type": "binary", "expected_values": [0, 1]},

        "GAME_NUMBER": {"type": "numeric"},
        "REGULAR_SEASON_GAME": {"type": "numeric"},
        "GAMES_REMAINING": {"type": "numeric"},
        "SEASON_PROGRESS": {"type": "numeric"},
        "IS_PLAYOFF": {"type": "binary", "expected_values": [0, 1]},
        "SEASON_END_PHASE": {"type": "binary", "expected_values": [0, 1]},

        "MONTH": {"type": "numeric"},
        "DAY_OF_WEEK": {"type": "numeric"},
        "WEEKEND": {"type": "binary", "expected_values": [0, 1]},
    }
    investigate_features(df, features)


def check_rolling_features(df):
    features = {
        "RECENT_MIN_3": {"type": "numeric"},
        "RECENT_MIN_5": {"type": "numeric"},
        "RECENT_MIN_10": {"type": "numeric"},
        "MIN_TREND": {"type": "numeric"},
        "WORKLOAD_SPIKE": {"type": "binary", "expected_values": [0, 1]},
        "MAX_MIN_5": {"type": "numeric"},
        "CONSISTENT_HEAVY": {"type": "binary", "expected_values": [0, 1]},
        "RECENT_PERFORMANCE": {"type": "numeric"},

        "TEAM_WIN_RATE_10": {"type": "numeric"},
        "TEAM_WINNING_STREAK": {"type": "binary", "expected_values": [0, 1]},
        "TEAM_LOSING_STREAK": {"type": "binary", "expected_values": [0, 1]},
    }
    investigate_features(df, features)


def check_opponent_features(df):
    features = {
        "OPPONENT_STRENGTH": {"type": "numeric"},
        "STRONG_OPPONENT": {"type": "binary", "expected_values": [0, 1]},
        "WEAK_OPPONENT": {"type": "binary", "expected_values": [0, 1]},
        "RATING_DIFF": {"type": "numeric"},
        "FAVORABLE_MATCHUP": {"type": "binary", "expected_values": [0, 1]},
        "TOUGH_MATCHUP": {"type": "binary", "expected_values": [0, 1]},
    }
    investigate_features(df, features)


def check_team_features(df):
    features = {
        "TEAM_OFFENSE": {"type": "numeric"},
        "HIGH_OFFENSE_TEAM": {"type": "binary", "expected_values": [0, 1]},
        "OPPONENT_OFFENSE": {"type": "numeric"},
    }
    investigate_features(df, features)


def check_injury_features(df):
    features = {
        "IS_INJURED": {"type": "binary", "expected_values": [0, 1]},
        "PREV_INJURED": {"type": "binary", "expected_values": [0, 1]},
        "NEW_INJURY_EVENT": {"type": "binary", "expected_values": [0, 1]},
        "INJURY_COUNT_SEASON": {"type": "numeric"},
        "HAS_INJURY_HISTORY": {"type": "binary", "expected_values": [0, 1]},
        "INJURED_NEXT_GAME": {"type": "binary", "expected_values": [0, 1]},

        "FATIGUE_RISK": {"type": "binary", "expected_values": [0, 1]},
        "B2B_HEAVY_RISK": {"type": "binary", "expected_values": [0, 1]},
        "PURE_FATIGUE_RISK": {"type": "numeric"},
        "ANY_FATIGUE": {"type": "binary", "expected_values": [0, 1]},

        "POOR_CONDITION": {"type": "binary", "expected_values": [0, 1]},
        "CONDITION": {"type": "numeric"},
        "INJURY_HISTORY_INDEX": {"type": "numeric"},
    }
    investigate_features(df, features)


def check_physical_features(df):
    features = {
        "BMI": {"type": "numeric"},
        "AGE_GROUP": {"type": "numeric"},  # encoded as 0,1,2,3 (float)
        "IS_GUARD": {"type": "binary", "expected_values": [0, 1]},
        "IS_FORWARD": {"type": "binary", "expected_values": [0, 1]},
        "IS_CENTER": {"type": "binary", "expected_values": [0, 1]},
        "IS_BIG": {"type": "binary", "expected_values": [0, 1]},
        "HEAVY_PLAYER": {"type": "binary", "expected_values": [0, 1]},
    }
    investigate_features(df, features)


df = load_data()

check_basic_features(df)
check_time_features(df)
check_players_features(df)
check_rolling_features(df)
check_opponent_features(df)
check_team_features(df)
check_injury_features(df)
check_physical_features(df)
