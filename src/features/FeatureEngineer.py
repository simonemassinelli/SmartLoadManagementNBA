import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop('Unnamed: 0', axis = 1)

        self.df['GAME_DATE_EST'] = pd.to_datetime(self.df['GAME_DATE_EST'])

        numeric_cols = [
            'SEASON', 'DRAFT_YEAR', 'AGE', 'PLAYER_HEIGHT','PLAYER_WEIGHT', 'GP', 'MIN_INT', 'RATING', 'PLUS_MINUS', 'OREB_PCT', 'DREB_PCT', 'USG_PCT', 'PTS_home', 'PTS_away', 'POINTS_DIFF', 'CONDITION', 'INJURY_HISTORY_INDEX', 'HOME_GAME', 'IS_INJURED'
        ]

        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        if 'SEASON' in self.df.columns:
            self.df['SEASON'] = self.df['SEASON'].apply(
                lambda x: x + 2000 if x < 100 else x
            )

        self.df = self.df.sort_values(['PLAYER_NAME', 'GAME_DATE_EST'])

        print(f"{len(self.df)} rows")
        print(f"Unique players: {self.df['PLAYER_NAME'].nunique()}")
        print(f"Unique teams: {self.df['PLAYER_TEAM'].nunique()}")

    def engineer_all_features(self):
        self.basic_features()
        self.players_features()
        self.time_features()
        self.rolling_features()
        self.opponent_features()
        self.team_features()
        self.injury_features()
        self.physical_features()

        self.cleanup()

        print(f"Total features: {len(self.df.columns)}")
        print(f"Total rows: {len(self.df)}")

        return self.df


    def basic_features(self):
        self.df['TEAM_WON'] = np.where(
            self.df['HOME_GAME'] == 1,
            (self.df['PTS_home'] > self.df['PTS_away']).astype(int),
            (self.df['PTS_away'] > self.df['PTS_home']).astype(int)
        )

        self.df['TOTAL_POINTS'] = self.df['PTS_home'] + self.df['PTS_away']
        self.df['HIGH_SCORING'] = (self.df['TOTAL_POINTS'] > 200).astype(int)
        self.df['BLOWOUT'] = (abs(self.df['POINTS_DIFF']) > 15).astype(int)
        self.df['CLOSE_GAME'] = (abs(self.df['POINTS_DIFF']) < 5).astype(int)
        self.df['LOW_USAGE'] = (self.df['MIN_INT'] < 10).astype(int)
        self.df['DNP'] = (self.df['MIN_INT'] == 0).astype(int) # did not play

    def players_features(self):
        self.df['YEARS_IN_LEAGUE'] = (
            self.df['SEASON'] - self.df['DRAFT_YEAR']
        ).clip(lower = 0)


        #  Mark rookie only for the player's first game. Alternative: mark all games from the player's first season as rookie (?)
        self.df['IS_ROOKIE'] = (self.df.groupby('PLAYER_NAME').cumcount() == 0).astype(int)


        self.df['IS_VETERAN'] = (self.df['YEARS_IN_LEAGUE'] >= 5).astype(int)
        self.df['AVAILABILITY'] = (self.df['GP'] / 82.0).clip(upper = 1.0)

    def time_features(self):
        self.df['DAYS_REST'] = (
            self.df.groupby('PLAYER_NAME')['GAME_DATE_EST']
            .diff()
            .dt.days
            .fillna(7)
        )

        self.df['BACK_TO_BACK'] = (self.df['DAYS_REST'] == 1).astype(int)
        # Not sure about the number of days
        self.df['WELL_RESTED'] = (self.df['DAYS_REST'] >= 3).astype(int)

        team_games = (
            self.df[['SEASON', 'PLAYER_TEAM', 'GAME_DATE_EST']]
            .drop_duplicates()
            .sort_values(['SEASON', 'PLAYER_TEAM', 'GAME_DATE_EST'])
            .copy()
        )
        team_games['GAME_NUMBER'] = (
            team_games
            .groupby(['SEASON', 'PLAYER_TEAM'])
            .cumcount() + 1
        )
        self.df = self.df.merge(
            team_games,
            on = ['SEASON', 'PLAYER_TEAM', 'GAME_DATE_EST'],
            how = 'left',
            suffixes = ('', '_tg')
        )


        regular_season_game = self.df['GAME_NUMBER'].clip(upper = 82)
        self.df['GAMES_REMAINING'] = (82 - regular_season_game).astype(int)
        self.df['SEASON_PROGRESS'] = regular_season_game / 82.0
        self.df['REGULAR_SEASON_GAME'] = regular_season_game
        self.df['IS_PLAYOFF'] = (self.df['GAME_NUMBER'] > 82).astype(int)

        self.df['SEASON_END_PHASE'] = (self.df['GAME_NUMBER'] >= 66).astype(int)

        self.df['MONTH'] = self.df['GAME_DATE_EST'].dt.month
        self.df['DAY_OF_WEEK'] = self.df['GAME_DATE_EST'].dt.dayofweek
        self.df['WEEKEND'] = (self.df['DAY_OF_WEEK'] >= 5).astype(int)

    def rolling_features(self):

        # How many mins a player has been playing recently (3, 5, 10) games
        self.df['RECENT_MIN_3'] = (
            self.df.groupby('PLAYER_NAME')['MIN_INT']
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )

        self.df['RECENT_MIN_5'] = (
            self.df.groupby('PLAYER_NAME')['MIN_INT']
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )

        self.df['RECENT_MIN_10'] = (
            self.df.groupby('PLAYER_NAME')['MIN_INT']
            .rolling(window=10, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )

        self.df['MIN_TREND'] = self.df['RECENT_MIN_3'] - self.df['RECENT_MIN_5']

        # if current much higher than recent avg
        self.df['WORKLOAD_SPIKE'] = (
            (self.df['MIN_INT'] - self.df['RECENT_MIN_5']) > 10
        ).astype(int)

        # max mins in last 5 games
        self.df['MAX_MIN_5'] = (
            self.df.groupby('PLAYER_NAME')['MIN_INT']
            .rolling(window=5, min_periods = 1)
            .max()
            .reset_index(drop=True)
        )

        # consistent heavy workload
        self.df['CONSISTENT_HEAVY'] = (
            self.df['RECENT_MIN_5'] > 35
        ).astype(int)

        self.df['RECENT_PERFORMANCE'] = (
            self.df.groupby('PLAYER_NAME')['PLUS_MINUS']
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )

        team_results = (
            self.df[['SEASON', 'PLAYER_TEAM', 'GAME_DATE_EST', 'TEAM_WON']]
            .drop_duplicates()
            .sort_values(['SEASON', 'PLAYER_TEAM', 'GAME_DATE_EST'])
            .copy()
        )

        team_results['TEAM_WIN_RATE_10'] = (
            team_results
            .groupby(['SEASON', 'PLAYER_TEAM'])['TEAM_WON']
            .rolling(window=10, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        self.df = self.df.merge(
            team_results[['SEASON', 'PLAYER_TEAM', 'GAME_DATE_EST', 'TEAM_WIN_RATE_10']],
            on=['SEASON', 'PLAYER_TEAM', 'GAME_DATE_EST'],
            how='left'
        )

        self.df['TEAM_WINNING_STREAK'] = (
                self.df['TEAM_WIN_RATE_10'] > 0.6
        ).astype(int)

        self.df['TEAM_LOSING_STREAK'] = (
                self.df['TEAM_WIN_RATE_10'] < 0.4
        ).astype(int)

    def opponent_features(self):
        opponent_ratings = (
            self.df.groupby('AWAY_TEAM')['RATING']
            .mean()
            .to_dict()
        )

        self.df['OPPONENT_STRENGTH'] = np.where(
            self.df['HOME_GAME'] == 1,
            self.df['AWAY_TEAM'].map(opponent_ratings),
            self.df['HOME_TEAM'].map(opponent_ratings)
        ).astype(float)

        self.df['OPPONENT_STRENGTH'] = self.df['OPPONENT_STRENGTH'].fillna(self.df['OPPONENT_STRENGTH'].mean())

        mean_strength = self.df['OPPONENT_STRENGTH'].mean()
        std_strength = self.df['OPPONENT_STRENGTH'].std()

        self.df['STRONG_OPPONENT'] = (
            self.df['OPPONENT_STRENGTH'] > (mean_strength + 0.5 * std_strength)
        ).astype(int)

        self.df['WEAK_OPPONENT'] = (
            self.df['OPPONENT_STRENGTH'] < (mean_strength - 0.5 * std_strength)
        ).astype(int)

        self.df['RATING_DIFF'] = (
            self.df['RATING'] - self.df['OPPONENT_STRENGTH']
        )

        favorable_threshold = self.df['RATING_DIFF'].quantile(0.67)
        tough_threshold = self.df['RATING_DIFF'].quantile(0.33)

        self.df['FAVORABLE_MATCHUP'] = (
            self.df['RATING_DIFF'] > favorable_threshold
        ).astype(int)

        self.df['TOUGH_MATCHUP'] = (
            self.df['RATING_DIFF'] < tough_threshold
        ).astype(int)


    def team_features(self):

        # offensive rating of the team (avg points)
        team_offense = (
            self.df.groupby('PLAYER_TEAM')
            .apply(lambda x : x[x['HOME_GAME'] == 1]['PTS_home'].mean()
                   if len(x[x['HOME_GAME'] == 1]) > 0
                   else x['PTS_away'].mean(),
                   include_groups = False)
            .to_dict()
        )

        self.df['TEAM_OFFENSE'] = (
            self.df['PLAYER_TEAM'].map(team_offense).fillna(100)
        )
        self.df['HIGH_OFFENSE_TEAM'] = (
            self.df['TEAM_OFFENSE'] > 110
        ).astype(int)

        opponent_offense = (
            self.df.groupby('AWAY_TEAM')
            .apply(lambda x: x['PTS_away'].mean(),
                   include_groups = False)
            .to_dict()
        )

        self.df['OPPONENT_OFFENSE'] = np.where(
            self.df['HOME_GAME'] == 1,
            self.df['AWAY_TEAM'].map(opponent_offense),
            self.df['HOME_TEAM'].map(opponent_offense)
        ).astype(float)

        self.df['OPPONENT_OFFENSE'] = self.df['OPPONENT_OFFENSE'].fillna(100)

    def injury_features(self):
        self.df['PREV_INJURED'] = (
            self.df.groupby('PLAYER_NAME')['IS_INJURED']
            .shift(1)
            .fillna(0)
        )

        self.df['NEW_INJURY_EVENT'] = (
            (self.df.groupby(['PLAYER_NAME', 'SEASON'])['IS_INJURED']
             .diff() == 1)
            .astype(int)
        )

        self.df['INJURY_COUNT_SEASON'] = (
            self.df.groupby(['PLAYER_NAME', 'SEASON'])['NEW_INJURY_EVENT']
            .cumsum()
        )

        self.df['HAS_INJURY_HISTORY'] = (
            self.df['INJURY_COUNT_SEASON'] > 0
        ).astype(int)

        self.df['INJURED_NEXT_GAME'] = (
            self.df.groupby('PLAYER_NAME')['IS_INJURED'].shift(-1).fillna(0)
        )

        self.df['FATIGUE_RISK'] = (
            (self.df['MIN_INT'] > 35) & (self.df['USG_PCT'] > 0.25)
        ).astype(int)

        # Back to back heavy risk
        self.df['B2B_HEAVY_RISK'] = (
                (self.df['BACK_TO_BACK'] == 1) & (self.df['MIN_INT'] > 35)
        ).astype(int)

        self.df['PURE_FATIGUE_RISK'] = (
                self.df['FATIGUE_RISK'] + self.df['B2B_HEAVY_RISK']
        )
        self.df['ANY_FATIGUE'] = (self.df['PURE_FATIGUE_RISK'] > 0).astype(int)

        self.df['POOR_CONDITION'] = (
            self.df['CONDITION'] < 85
        ).astype(int)


    def physical_features(self):
        height_m = self.df['PLAYER_HEIGHT'] / 100
        self.df['BMI'] = self.df['PLAYER_WEIGHT'] / (height_m ** 2) # Body mass index

        self.df['AGE_GROUP'] = pd.cut(
            self.df['AGE'],
            bins = [0, 25, 30, 35, 100],
            labels = [0, 1, 2, 3]
        ).astype(float)

        self.df['IS_GUARD'] = self.df['POS'].isin(['PG', 'SG']).astype(int)
        self.df['IS_FORWARD'] = self.df['POS'].isin(['SF', 'PF']).astype(int)
        self.df['IS_CENTER'] = (self.df['POS'] == 'C').astype(int)
        self.df['IS_BIG'] = self.df['POS'].isin(['C', 'PF']).astype(int)
        self.df['HEAVY_PLAYER'] = (
            self.df['PLAYER_WEIGHT'] > 110
        ).astype(int)

    def cleanup(self):
        numeric_cols = self.df.select_dtypes(include = [np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)

        self.df = self.df.replace([np.inf, -np.inf], 0)
        self.df = self.df.dropna(subset = ['PLAYER_NAME', 'GAME_DATE_EST'])




