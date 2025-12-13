import torch
import numpy as np

AVG_INJURY_GAMES = 7  # average NBA injury duration in games


def optimize_minutes_today(model, game_input, candidate_minutes_list, n_simulations=1000, device='cpu'):
    """
    Optimize minutes allocation considering injury risks over average injury duration.

    Parameters:
    - model: trained SmartLoadModel
    - game_input: dictionary of single game features (player slots padded)
    - candidate_minutes_list: list of arrays, each array is candidate minutes allocation
    - n_simulations: number of Monte Carlo simulations
    - device: 'cpu' or 'cuda'

    Returns:
    - best_minutes: minutes allocation that maximizes objective
    - best_score: objective value
    """
    model.eval()
    game_batch_template = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
    if isinstance(v, np.ndarray) else v for k, v in game_input.items()}

    best_score = -np.inf
    best_minutes = None

    for minutes in candidate_minutes_list:
        game_batch_template['MIN_INT'] = torch.tensor(minutes, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            preds_today = model(game_batch_template)
            win_prob_today = torch.sigmoid(preds_today["win_logits"])[0, 0].item()
            injury_probs = torch.sigmoid(preds_today["injury_logits"])[0].cpu().numpy()

        # simulate injuries over next AVG_INJURY_GAMES
        future_win_probs = []
        for _ in range(n_simulations):
            injured_next_games = np.zeros(len(injury_probs))
            # simulate whether each player gets injured today
            injured_today = np.random.rand(len(injury_probs)) < injury_probs
            injured_next_games += injured_today.astype(int) * AVG_INJURY_GAMES

            # construct masked input for future games (simplified: same as today)
            future_win = 0
            for g in range(AVG_INJURY_GAMES):
                future_mask = (injured_next_games <= g).astype(float)  # 1 if available
                future_game_input = game_input.copy()
                future_game_input['player_mask'] = future_mask
                future_game_batch = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
                if isinstance(v, np.ndarray) else v for k, v in future_game_input.items()}
                with torch.no_grad():
                    fut_preds = model(future_game_batch)
                    future_win += torch.sigmoid(fut_preds["win_logits"])[0, 0].item()

            future_win_probs.append(future_win / AVG_INJURY_GAMES)

        expected_future_win = np.mean(future_win_probs)
        objective = win_prob_today + expected_future_win

        if objective > best_score:
            best_score = objective
            best_minutes = minutes

    return best_minutes, best_score