from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class GameDataset(Dataset):
    SHARED_FEATURES = [
        'HOME_GAME', 'BACK_TO_BACK', 'OPPONENT_STRENGTH',
        'GAMES_REMAINING', 'SEASON_END_PHASE', 'SEASON_PROGRESS',
        'TEAM_WIN_RATE_10', 'TEAM_WINNING_STREAK', 'TEAM_LOSING_STREAK',
        'HIGH_SCORING', 'BLOWOUT', 'CLOSE_GAME',
        'MONTH', 'DAY_OF_WEEK', 'WEEKEND', 'IS_PLAYOFF',
        'TEAM_OFFENSE', 'OPPONENT_OFFENSE', 'HIGH_OFFENSE_TEAM',
        'REGULAR_SEASON_GAME'
    ]

    PLAYER_FEATURES = [
        'AGE', 'PLAYER_HEIGHT', 'PLAYER_WEIGHT', 'YEARS_IN_LEAGUE',
        'RATING', 'USG_PCT', 'NET_RATING_real', 'OREB_PCT', 'DREB_PCT',
        'PLAYER_IMPORTANCE', 'START_RATIO', 'AVAILABILITY',
        'IS_GUARD', 'IS_FORWARD', 'IS_CENTER', 'IS_BIG', 'BMI',
        'IS_ROOKIE', 'IS_VETERAN', 'IS_STARTER'
    ]

    WIN_FEATURES = [
        'TEAM_OFFENSE', 'OPPONENT_OFFENSE', 'TEAM_WIN_RATE_10',
        'OPPONENT_STRENGTH', 'STRONG_OPPONENT', 'WEAK_OPPONENT',
        'HOME_GAME', 'SEASON_END_PHASE', 'IS_PLAYOFF', 'RATING_DIFF'
    ]

    INJURY_FEATURES = [
        'AGE', 'RECENT_MIN_5', 'BACK_TO_BACK', 'PREV_INJURED',
        'FATIGUE_RISK', 'B2B_HEAVY_RISK', 'MIN_TREND',
        'WORKLOAD_SPIKE', 'POOR_CONDITION', 'INJURY_HISTORY_INDEX',
        'HAS_INJURY_HISTORY', 'CONDITION', 'ANY_FATIGUE',
        'PURE_FATIGUE_RISK', 'YEARS_IN_LEAGUE'
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


    def _prepare_games(self):
        games = []

        grouped = self.df.groupby(['SEASON', 'PLAYER_TEAM', 'GAME_DATE_EST'])

        for (season, team, date), group in grouped:
            if len(group) == 0:
                continue

            group = group.sort_values('MIN_INT', ascending = False)
            games.append({
                'season' : season,
                'team' : team,
                'date' : date,
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

        win_feats = game_data[self.WIN_FEATURES].iloc[0].value.astype(np.float32)

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


