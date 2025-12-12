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
        'HOME_TEAM_POSITION',
        'HOME_TEAM_W_PCT',
        'AWAY_TEAM_POSITION',
        'AWAY_TEAM_W_PCT'
    ]

    PLAYER_FEATURES = [
        'AGE',
        'PLAYER_HEIGHT',
        'PLAYER_WEIGHT',
        'BMI',
        'AGE_GROUP',
        'YEARS_IN_LEAGUE',
        'IS_ROOKIE',
        'IS_VETERAN',
        'RATING',
        'NET_RATING_real',
        'USG_PCT',
        'OREB_PCT',
        'DREB_PCT',
        'PLAYER_IMPORTANCE',
        'START_RATIO',
        'AVAILABILITY',
        'IS_STARTER',
        'IS_GUARD',
        'IS_FORWARD',
        'IS_CENTER',
        'IS_BIG',
        'HEAVY_PLAYER',
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
        'DAYS_REST',
        'DRAFT_YEAR',
        'GP_real'
    ]

    WIN_FEATURES = [
        'HOME_GAME',
        'TEAM_WIN_RATE_10',
        'TEAM_WINNING_STREAK',
        'TEAM_LOSING_STREAK',
        'TEAM_OFFENSE',
        'OPPONENT_OFFENSE',
        'HIGH_OFFENSE_TEAM',
        'OPPONENT_STRENGTH',
        'STRONG_OPPONENT',
        'WEAK_OPPONENT',
        'RATING_DIFF',
        'SEASON_END_PHASE',
        'IS_PLAYOFF'
    ]

    INJURY_FEATURES = [
        'AGE',
        'BMI',
        'YEARS_IN_LEAGUE',
        'DAYS_REST',
        'BACK_TO_BACK',
        'WELL_RESTED',
        'PREV_INJURED',
        'INJURY_HISTORY_INDEX',
        'HAS_INJURY_HISTORY',
        'CONDITION',
        'POOR_CONDITION',
        'MIN_INT_LAG1',
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
        'AGE_GROUP',
        'INJURY_COUNT_SEASON',
        'IS_INJURED',
        'LOW_USAGE_LAG1'
    ]

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        players_per_game = (
            self.df.groupby(['SEASON', 'PLAYER_TEAM', 'GAME_DATE_EST'])
            .size()
        )
        self.max_players = int(players_per_game.max()) + 1
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


    def count_features(self):
        return len(self.PLAYER_FEATURES) + len(self.INJURY_FEATURES) + len(self.SHARED_FEATURES) + len(self.WIN_FEATURES)


    def _prepare_games(self):
        games = []

        grouped = self.df.groupby(['SEASON', 'PLAYER_TEAM', 'GAME_DATE_EST'])

        for (season, team, date), group in grouped:
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

        won = game_data['TEAM_WON'].iloc[0]

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


