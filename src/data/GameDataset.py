from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class GameDataset(Dataset):
    SHARED_FEATURES = [
        'HOME_GAME',
        'BACK_TO_BACK',
        'WELL_RESTED',
        'ROAD_GAMES_STREAK',
        'GAMES_REMAINING',
        'SEASON_PROGRESS',
        'SEASON_END_PHASE',
        'REGULAR_SEASON_GAME',
        'IS_PLAYOFF',
        'MONTH',
        'DAY_OF_WEEK',
        'WEEKEND',
        'TEAM_WIN_RATE_10',
        'TEAM_WINNING_STREAK',
        'TEAM_LOSING_STREAK',
        'TEAM_OFFENSE',
        'OPPONENT_OFFENSE',
        'HIGH_OFFENSE_TEAM',
        'OPPONENT_STRENGTH',
        'STRONG_OPPONENT',
        'WEAK_OPPONENT',
        'HOME_TEAM_W_PCT',
        'AWAY_TEAM_W_PCT',
        'TEAM_POSITION',
        'OPPONENT_POSITION',
    ]

    PLAYER_FEATURES = [
        'AGE',
        'PLAYER_HEIGHT',
        'PLAYER_WEIGHT',
        'BMI',
        'AGE_GROUP',
        'HEAVY_PLAYER',
        'YEARS_IN_LEAGUE',
        'IS_ROOKIE',
        'IS_VETERAN',
        'DRAFT_YEAR',
        'IS_GUARD',
        'IS_FORWARD',
        'IS_CENTER',
        'IS_BIG',
        'NET_RATING_real',
        'USG_PCT',
        'OREB_PCT',
        'DREB_PCT',
        'PLAYER_IMPORTANCE',
        'GP_real',
        'START_RATIO',
        'AVAILABILITY',
        'IS_STARTER',
        'DAYS_REST',
        'MIN_INT_LAG1',
        'PLUS_MINUS_LAG1',
        'LOW_USAGE_LAG1',
        'RECENT_MIN_3',
        'RECENT_MIN_5',
        'RECENT_MIN_10',
        'MIN_TREND',
        'WORKLOAD_SPIKE',
        'MAX_MIN_5',
        'CONSISTENT_HEAVY',
        'RECENT_PERFORMANCE',
        'RATING'
    ]

    WIN_FEATURES = [
        'HOME_GAME',
        'SEASON_END_PHASE',
        'IS_PLAYOFF',
        'BACK_TO_BACK',
        'TEAM_WIN_RATE_10',
        'TEAM_WINNING_STREAK',
        'TEAM_LOSING_STREAK',
        'TEAM_OFFENSE',
        'OPPONENT_OFFENSE',
        'HIGH_OFFENSE_TEAM',
        'OPPONENT_STRENGTH',
        'STRONG_OPPONENT',
        'WEAK_OPPONENT',
        'TEAM_STRENGTH_DIFF',
        'FAVORABLE_MATCHUP',
        'TOUGH_MATCHUP',
        'HOME_TEAM_W_PCT',
        'AWAY_TEAM_W_PCT',
        'TEAM_POSITION',
        'OPPONENT_POSITION',
    ]

    INJURY_FEATURES = [
        'AGE',
        'BMI',
        'AGE_GROUP',
        'HEAVY_PLAYER',
        'YEARS_IN_LEAGUE',
        'DAYS_REST',
        'BACK_TO_BACK',
        'WELL_RESTED',
        'PREV_INJURED',
        'INJURY_HISTORY_INDEX',
        'HAS_INJURY_HISTORY',
        'INJURY_COUNT_SEASON',
        'INJURY_NEARBY_PAST',
        'CONDITION',
        'POOR_CONDITION',
        'MIN_INT_LAG1',
        'LOW_USAGE_LAG1',
        'RECENT_MIN_3',
        'RECENT_MIN_5',
        'RECENT_MIN_10',
        'MIN_TREND',
        'WORKLOAD_SPIKE',
        'MAX_MIN_5',
        'CONSISTENT_HEAVY',
        'FATIGUE_RISK_LAG1',
        'B2B_HEAVY_RISK_LAG1',
        'PURE_FATIGUE_RISK_LAG1',
        'ANY_FATIGUE_LAG1',
        'IS_INJURED' # ?
    ]

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        self._verify_columns()

        self.df['GAME_DATE_EST'] = pd.to_datetime(self.df['GAME_DATE_EST'], errors='coerce')

        players_per_game = (
            self.df.groupby(['SEASON', 'PLAYER_TEAM', 'GAME_DATE_EST', 'GAME_ID'])
            .size()
        )
        self.max_players = int(players_per_game.max())
        print(f"Max players: {self.max_players}")

        self.games = self._prepare_games()
        print("Dataset ready")
        print(f"Total games: {len(self.games):,}")
        print(f"Features")
        print(f"Shared: {len(self.SHARED_FEATURES)}")
        print(f"Player: {len(self.PLAYER_FEATURES)}")
        print(f"Win: {len(self.WIN_FEATURES)}")
        print(f"Injury:  {len(self.INJURY_FEATURES)}")
        print(f"Total features: {self.count_features()}")
        print(f"Total unique features: {self.count_unique_features()}")


    def count_features(self):
        return len(self.PLAYER_FEATURES) + len(self.INJURY_FEATURES) + len(self.SHARED_FEATURES) + len(self.WIN_FEATURES)

    def count_unique_features(self):
        return len(set(self.PLAYER_FEATURES + self.INJURY_FEATURES + self.SHARED_FEATURES + self.WIN_FEATURES))


    def _prepare_games(self):
        games = []

        grouped = self.df.groupby(['SEASON', 'PLAYER_TEAM', 'GAME_DATE_EST', 'GAME_ID'])

        for (season, team, date, game_id), group in grouped:
            if len(group) == 0:
                continue

            sort_col = 'MIN_INT_LAG1' if 'MIN_INT_LAG1' in group.columns else 'PLAYER_IMPORTANCE'

            group = group.sort_values(sort_col, ascending = False)
            games.append({
                'season' : season,
                'team' : team,
                'date' : pd.to_datetime(date),
                'data' : group.reset_index(drop = True)
            })
        return games


    def __len__(self):
        return len(self.games)


    def __getitem__(self, item):
        game = self.games[item]
        game_data = game['data']
        num_players = len(game_data)

        shared = game_data[self.SHARED_FEATURES].iloc[0].values.astype(np.float32)
        player_feats = np.zeros((self.max_players, len(self.PLAYER_FEATURES)), dtype = np.float32)
        player_feats[:num_players] = game_data[self.PLAYER_FEATURES].values

        actual_minutes = np.zeros(self.max_players, dtype = np.float32)
        actual_minutes[:num_players] = game_data['MIN_INT'].values

        player_mask = np.zeros(self.max_players, dtype = np.float32)
        player_mask[:num_players] = 1

        win_feats = game_data[self.WIN_FEATURES].iloc[0].values.astype(np.float32)

        injury_feats = np.zeros((self.max_players, len(self.INJURY_FEATURES)), dtype = np.float32)
        injury_feats[:num_players] = game_data[self.INJURY_FEATURES].values

        won = float(game_data['TEAM_WON'].iloc[0])

        injuries = np.zeros(self.max_players, dtype = np.float32)
        if 'INJURED_NEXT_GAME' in game_data.columns:
            injuries[:num_players]  = game_data['INJURED_NEXT_GAME'].values
        else:
            injuries[:num_players] = game_data['IS_INJURED'].values

        return{
            'shared_features': torch.FloatTensor(shared),
            'player_features': torch.FloatTensor(player_feats),
            'actual_minutes': torch.FloatTensor(actual_minutes),
            'player_mask': torch.FloatTensor(player_mask),
            'win_features': torch.FloatTensor(win_feats),
            'injury_features': torch.FloatTensor(injury_feats),
            'won': torch.FloatTensor([won]),
            'injuries': torch.FloatTensor(injuries),
            'num_players': num_players,
            'season': game['season'],
            'team': game['team'],
            'date': game['date'],
        }

    def _verify_columns(self):
        all_features = set()
        all_features.update(self.SHARED_FEATURES)
        all_features.update(self.PLAYER_FEATURES)
        all_features.update(self.WIN_FEATURES)
        all_features.update(self.INJURY_FEATURES)
        all_features.update(['MIN_INT', 'TEAM_WON', 'SEASON', 'PLAYER_TEAM', 'GAME_DATE_EST'])
        all_features.update(['GAME_ID'])
        missing = [feature for feature in all_features if feature not in self.df.columns]

        if missing:
            print(f"Missing {len(missing)} features")
            for col in sorted(missing):
                print(f"{col}")
            raise ValueError('Missing required columns')

