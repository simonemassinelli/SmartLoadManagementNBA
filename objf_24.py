import numpy as np
import torch
import copy

AVG_INJURY_GAMES = 7


def generate_initial_population(num_players, max_players, n_samples=50, step=4, total_minutes=240):
    """Generates random valid minute distributions summing to total_minutes."""
    candidates = []
    n_starters = min(5, num_players)
    n_bench = num_players - n_starters

    for _ in range(n_samples):
        minutes = np.zeros(num_players)

        # Random initialization
        starter_mins = np.random.randint(28 // step, 40 // step + 1, n_starters) * step
        bench_mins = np.random.randint(0 // step, 20 // step + 1, n_bench) * step

        minutes[:n_starters] = starter_mins
        if n_bench > 0:
            minutes[n_starters:] = bench_mins

        # Adjust to match exactly total_minutes
        diff = total_minutes - minutes.sum()
        while diff != 0:
            idx = np.random.randint(0, num_players)
            if diff > 0 and minutes[idx] + step <= 44:
                minutes[idx] += step
                diff -= step
            elif diff < 0 and minutes[idx] - step >= 0:
                minutes[idx] -= step
                diff += step

        padded = np.zeros(max_players)
        padded[:num_players] = minutes
        candidates.append(padded)

    return candidates


def mutate_minutes(minutes, num_players, step=4):
    """Swaps minutes between two players to explore new solutions."""
    new_mins = minutes.copy()
    idx1, idx2 = np.random.choice(num_players, 2, replace=False)

    if new_mins[idx1] >= step and new_mins[idx2] + step <= 44:
        new_mins[idx1] -= step
        new_mins[idx2] += step

    return new_mins


def evaluate_lineup(model, game_input, minutes_array, n_simulations, avg_injury_games, device):
    """Calculates objective: Win Probability Today + Expected Future Wins."""
    current_input = copy.deepcopy(game_input)
    current_input["MIN_INT"][0] = torch.tensor(minutes_array, dtype=torch.float32, device=device)

    with torch.no_grad():
        preds_today = model(current_input)
        win_prob_today = torch.sigmoid(preds_today["win_logits"])[0, 0].item()
        injury_probs = torch.sigmoid(preds_today["injury_logits"])[0].cpu().numpy()

    future_win_totals = []

    for _ in range(n_simulations):
        # Simulate injuries based on today's risk
        is_injured = np.random.rand(len(injury_probs)) < injury_probs
        injured_games_left = is_injured.astype(int) * avg_injury_games
        simulated_future_wins = 0.0

        for g in range(avg_injury_games):
            future_mask = (injured_games_left <= g).astype(float)
            current_input["player_mask"][0] = torch.tensor(future_mask, dtype=torch.float32, device=device)

            with torch.no_grad():
                future_preds = model(current_input)
                simulated_future_wins += torch.sigmoid(future_preds["win_logits"])[0, 0].item()

        future_win_totals.append(simulated_future_wins)

    expected_future_wins = np.mean(future_win_totals)
    return win_prob_today + expected_future_wins


def optimize_minutes_evolutionary(model, game_input, num_players, max_players,
                                  generations=20, population_size=40, n_sims_per_eval=50, device="cpu"):
    """Main Genetic Algorithm loop to optimize minutes."""
    print(f"Starting optimization: {generations} generations, population {population_size}...")

    candidates = generate_initial_population(num_players, max_players, n_samples=population_size)
    best_overall_score = -np.inf
    best_overall_minutes = None

    for gen in range(generations):
        scores = []

        # Evaluate population
        for mins in candidates:
            s = evaluate_lineup(model, game_input, mins, n_simulations=n_sims_per_eval,
                                avg_injury_games=AVG_INJURY_GAMES, device=device)
            scores.append(s)

            if s > best_overall_score:
                best_overall_score = s
                best_overall_minutes = mins.copy()

        # Selection (Elitism)
        sorted_indices = np.argsort(scores)[::-1]
        n_elite = population_size // 4
        top_indices = sorted_indices[:n_elite]
        top_candidates = [candidates[i] for i in top_indices]

        # Reproduction (Mutation)
        new_candidates = []
        new_candidates.extend(top_candidates)

        while len(new_candidates) < population_size:
            parent = top_candidates[np.random.randint(len(top_candidates))]
            child = mutate_minutes(parent, num_players, step=4)
            new_candidates.append(child)

        candidates = new_candidates

        if (gen + 1) % 5 == 0:
            print(f"Generation {gen + 1}/{generations} - Best Score: {best_overall_score:.4f}")

    return best_overall_minutes, best_overall_score