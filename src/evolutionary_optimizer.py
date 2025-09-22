# src/evolutionary_optimizer.py
import numpy as np
import random

def evolutionary_optimize(returns, risks, sentiments, population_size=50, generations=100, elite_frac=0.2, mutation_rate=0.1):
    """
    Evolutionary Algorithm for portfolio allocation.

    Args:
        returns (dict): {ticker: expected return}
        risks (dict): {ticker: annualized volatility}
        sentiments (dict): {ticker: sentiment score (-1, 0, +1)}
        population_size (int): number of candidate portfolios per generation
        generations (int): number of evolutionary steps
        elite_frac (float): fraction of top performers kept each gen
        mutation_rate (float): chance of small mutation in weights

    Returns:
        dict: {ticker: allocation weight}, best fitness score
    """
    tickers = list(returns.keys())
    n = len(tickers)

    #  weights 
    def normalize(weights):
        w = np.clip(weights, 0, None)
        return w / (w.sum() + 1e-9)

    # fitness function 
    def fitness(weights):
        # weights
        exp_return = sum(weights[i] * returns[t] for i, t in enumerate(tickers))
        risk = sum(weights[i] * risks[t] for i, t in enumerate(tickers))
        sentiment = sum(weights[i] * sentiments.get(t, 0) for i, t in enumerate(tickers))

        
        return exp_return - 0.5 * risk + 0.3 * sentiment

    # --- initialise population ---
    population = [normalize(np.random.rand(n)) for _ in range(population_size)]

    for _ in range(generations):
        scores = [fitness(ind) for ind in population]
        elite_count = max(1, int(elite_frac * population_size))

        # select elites
        elite_idx = np.argsort(scores)[-elite_count:]
        elites = [population[i] for i in elite_idx]

        # crossover + mutation
        children = []
        while len(children) < population_size - elite_count:
            p1, p2 = random.sample(elites, 2)
            cut = random.randint(1, n - 1)
            child = np.concatenate([p1[:cut], p2[cut:]])
            if random.random() < mutation_rate:
                idx = random.randint(0, n - 1)
                child[idx] += np.random.normal(0, 0.1) 
            children.append(normalize(child))

        population = elites + children

    # final evaluation
    scores = [fitness(ind) for ind in population]
    best = population[int(np.argmax(scores))]

    allocation = {t: round(float(w), 3) for t, w in zip(tickers, best)}
    return allocation, max(scores)
