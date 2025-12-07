import random
import numpy as np
from water_fill import water_fill
from middle import optimal_randomized_agent
from collections import deque

def setup_scenario(box_values, l, t):
    """
    Sets up the scenario with predefined boxes.
    Parameters:
        box_values (list): A list of integers representing the values in each box.
        l (int): Number of boxes the agent will select.
        t (int): Number of byzantine boxes the adversary can select.
    Returns:
        tuple: A tuple containing the n, l, t, and box_values.
    """

    # create a tuple representing the scenario
    n = len(box_values)
    return (n, l, t, box_values)


def simulate(scenario, agent_strategy, adversary_strategy, simulations, learning = False, history_size=5):
    """
    Simulates the agent and adversary picking boxes from the given scenario.
    Parameters:
        scenario (tuple): A tuple containing the n, l, t, and box_values.
        agent_strategy (function): Function defining the agent's strategy.
        adversary_strategy (function): Function defining the adversary's strategy.
        simulations (int): Number of simulations to run.
    Returns:
        total_agent_utility (int): Total utility gained by the agent.
        total_adversary_utility (int): Total utility gained by the adversary (lost by the agent).
    """

    # Total agent utility (agent gets the box value as the utility if not foiled by adversary)
    total_agent_utility = 0
    # Total adversary utility (adversary gets the box value as the utility if it foils the agent)
    total_adversary_utility = 0

    # Unpack the scenario
    n, l, t, boxes = scenario

    # Store a history for the learning agent
    history = deque([(0, 0)] * history_size, maxlen=history_size)

    # Run each simulation
    for i in range(simulations):
        # Agent picks boxes based on its strategy
        if learning:
            agent_chosen = agent_strategy(n, l, t, boxes, history)
        else:
            agent_chosen = agent_strategy(n, l, t, boxes)
        # Adversary picks boxes based on its strategy
        # Pass agent_strategy to optimal_byzantine_adversary so it knows the agent's distribution
        if adversary_strategy.__name__ == 'optimal_byzantine_adversary':
            adversary_chosen = adversary_strategy(n, l, t, boxes, agent_strategy)
        else:
            adversary_chosen = adversary_strategy(n, l, t, boxes)

        current_agent_utility = 0
        current_adversary_utility = 0

        # Calculate utilities for every box
        for j in range(len(agent_chosen)):
            # if the agent chose that box
            if agent_chosen[j][1] == 1:
                # check if the adversary also chose that box
                if adversary_chosen[j][1] == 1:
                    # adversary foiled the agent
                    current_adversary_utility += agent_chosen[j][0]
                    total_adversary_utility += agent_chosen[j][0]
                else:
                    # agent gets the box value as utility because he was not foiled
                    current_agent_utility += agent_chosen[j][0]
                    total_agent_utility += agent_chosen[j][0]

        #Update total utility per round, and use this in our history
        history.append((current_agent_utility, current_adversary_utility))

    # Compute average utilities
    total_agent_utility /= simulations
    total_adversary_utility /= simulations

    return total_agent_utility, total_adversary_utility

# =================== Agent Strategies ===================

def pick_randomly_agent(n, l, t, boxes):
    """
    Agent strategy: Picks boxes randomly.
    Parameters:
        n (int): Total number of boxes.
        l (int): Number of boxes the agent will select.
        t (int): Number of byzantine boxes the adversary can select.
        boxes (list): List of box values.
    Returns:
        list: A list of tuples where each tuple contains (box value, if chosen).
    """

    chosen_boxes = random.sample(range(n), l)
    return [(boxes[i], 1 if i in chosen_boxes else 0) for i in range(n)]

def deterministic_agent(n, l, t, boxes):
    """
    Agent strategy: Picks the top l boxes with the highest values.
    Parameters:
        n (int): Total number of boxes.
        l (int): Number of boxes the agent will select.
        t (int): Number of byzantine boxes the adversary can select.
        boxes (list): List of box values.
    Returns:
        list: A list of tuples where each tuple contains (box value, if chosen).
    """

    # Get the indexes of the top l boxes out of n total boxes
    top_indexes = sorted(range(n), key=lambda i: boxes[i], reverse=True)[:l]

    # Return the boxes with a flag (0 for no, 1 for yes) indicating if they were chosen
    return [(boxes[i], 1 if i in top_indexes else 0) for i in range(n)]

def greedy_agent(n, l, t, boxes):
    """
    Agent strategy: Picks random boxes from the top t + l boxes.
    Parameters:
        n (int): Total number of boxes.
        l (int): Number of boxes the agent will select.
        t (int): Number of byzantine boxes the adversary can select.
        boxes (list): List of box values.
    Returns:
        list: A list of tuples where each tuple contains (box value, if chosen).
    """

    # Get the indexes of the top t + l boxes out of n total boxes
    top_indexes = sorted(range(n), key=lambda i: boxes[i], reverse=True)[:(t + l)]
    chosen_boxes = random.sample(top_indexes, l)

    # Return the boxes with a flag (0 for no, 1 for yes) indicating if they were chosen
    return [(boxes[i], 1 if i in chosen_boxes else 0) for i in range(n)]


def safe_agent(n, l, t, boxes):
    """
    Agent strategy: Picks the l boxes immediately following the top t boxes.
    Parameters:
        n (int): Total number of boxes.
        l (int): Number of boxes the agent will select.
        t (int): Number of byzantine boxes the adversary can select.
        boxes (list): List of box values.
    Returns:
        list: A list of tuples where each tuple contains (box value, if chosen).
    """

    # Get the indexes of the top t + l boxes out of n total boxes
    top_indexes = sorted(range(n), key=lambda i: boxes[i], reverse=True)[:(t + l)]
    chosen_boxes = top_indexes[t:t + l]

    # Return the boxes with a flag (0 for no, 1 for yes) indicating if they were chosen
    return [(boxes[i], 1 if i in chosen_boxes else 0) for i in range(n)]


# Defines the water fill agent strategy (should be the best performing, lowest loss)
def water_fill_agent(n, l, t, boxes):
    # Water fill algorithm requires boxes sorted in descending order (v₁ ≥ v₂ ≥ ... ≥ vₙ) so this creates a list of (original_index, value) pairs and sort by value descending
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)
    
    # Extract sorted values and create mapping from sorted index to original index
    sorted_boxes = [val for _, val in indexed_boxes]
    index_map = [orig_idx for orig_idx, _ in indexed_boxes]
    
    # Call water_fill with sorted boxes
    max_val, optimal_p_prime = water_fill(sorted_boxes, t, l)
    
    # Map probabilities back to original indices because water_fill returns probabilities for sorted boxes
    # optimal_p_prime[i] is the probability for sorted box i, we need it for original box index_map[i]
    original_p_prime = [0.0] * n
    for sorted_idx in range(n):
        orig_idx = index_map[sorted_idx]
        original_p_prime[orig_idx] = optimal_p_prime[sorted_idx]
    
    # Use optimal_randomized_agent with original boxes and remapped probabilities
    return optimal_randomized_agent(n, l, t, boxes, original_p_prime)

# 

# =================== Adversary Strategies ===================
def pick_randomly_adversary(n, l, t, boxes):
    """
    Adversary strategy: Picks boxes randomly.
    Parameters:
        n (int): Total number of boxes.
        l (int): Number of boxes the agent will select.
        t (int): Number of byzantine boxes the adversary can select.
        boxes (list): List of box values.
    Returns:
        list: A list of tuples where each tuple contains (box value, if chosen).
    """

    chosen_boxes = random.sample(range(n), t)
    return [(boxes[i], 1 if i in chosen_boxes else 0) for i in range(n)]

def deterministic_adversary(n, l, t, boxes):
    """
    Adversary strategy: Picks the top t boxes with the highest values.
    Parameters:
        n (int): Total number of boxes.
        l (int): Number of boxes the agent will select.
        t (int): Number of byzantine boxes the adversary can select.
        boxes (list): List of box values.
    Returns:
        list: A list of tuples where each tuple contains (box value, if chosen).
    """

    # Get the indexes of the top t boxes out of n total boxes
    top_indexes = sorted(range(n), key=lambda i: boxes[i], reverse=True)[:t]

    # Return the boxes with a flag (0 for no, 1 for yes) indicating if they were chosen
    return [(boxes[i], 1 if i in top_indexes else 0) for i in range(n)]

def expected_value_adversary(n, l, t, boxes):
    """
    Adversary strategy: Picks boxes based on expected value calculations.

    Probabilistically picks boxes that maximize expected damage to the agent. 
    Each box is chosen with probability proportional to the expected utility it provides to the agent.
    Takes into account the agent's likely selections.

    Parameters:
        n (int): Total number of boxes.
        l (int): Number of boxes the agent will select.
        t (int): Number of byzantine boxes the adversary can select.
        boxes (list): List of box values.
    Returns:
        list: A list of tuples where each tuple contains (box value, if chosen).
    """

    # Convert to a numpy array to allow for no repeated selections
    np_boxes = np.array(boxes, dtype=float)

    # Determine the probability that the agent will pick each box (based on article and the fact that the agent wants a high utility)
    if np.sum(np_boxes) == 0:
        probabilities = np.ones(n) / n  # Avoid division by zero; uniform probabilities
    else:
        # Calculate probabilities proportional to box values
        probabilities = np_boxes / np.sum(np_boxes)

    # Select t boxes based on calculated probabilities, without replacement
    chosen_boxes = np.random.choice(n, size=t, replace=False, p=probabilities)

    # Return the boxes with a flag (0 for no, 1 for yes) indicating if they were chosen
    return [(boxes[i], 1 if i in chosen_boxes else 0) for i in range(n)]

def optimal_byzantine_adversary(n, l, t, boxes, agent_strategy=None):
    """
    Optimal adversary strategy: Knows the agent's distribution and nullifies
    the t boxes with the largest expected values (v_i * p_i).
    This implements the worst-case adversary from the paper.
    
    If agent_strategy is provided, computes that agent's distribution.
    Otherwise, assumes water fill (for backward compatibility).

    Parameters:
        n (int): Total number of boxes.
        l (int): Number of boxes the agent will select.
        t (int): Number of byzantine boxes the adversary can select.
        boxes (list): List of box values.
        agent_strategy (function, optional): Function defining the agent's strategy.
    Returns:
        list: A list of tuples where each tuple contains (box value, if chosen)
    """
    agent_distribution = [0.0] * n
    
    if agent_strategy is not None:
        strategy_name = agent_strategy.__name__
        
        # Handle deterministic strategies directly (no need to sample)
        if strategy_name == 'deterministic_agent':
            # Deterministic picks top l boxes: p_i = 1 for top l, 0 otherwise
            top_indexes = sorted(range(n), key=lambda i: boxes[i], reverse=True)[:l]
            for idx in top_indexes:
                agent_distribution[idx] = 1.0
        elif strategy_name == 'safe_agent':
            # Safe picks boxes t through t+l-1: p_i = 1 for those, 0 otherwise
            top_indexes = sorted(range(n), key=lambda i: boxes[i], reverse=True)[:(t + l)]
            chosen_boxes = top_indexes[t:t + l]
            for idx in chosen_boxes:
                agent_distribution[idx] = 1.0
        elif strategy_name == 'water_fill_agent':
            # Use water fill distribution
            indexed_boxes = list(enumerate(boxes))
            indexed_boxes.sort(key=lambda x: x[1], reverse=True)
            sorted_boxes = [val for _, val in indexed_boxes]
            index_map = [orig_idx for orig_idx, _ in indexed_boxes]
            max_val, optimal_p_prime = water_fill(sorted_boxes, t, l)
            for sorted_idx in range(n):
                orig_idx = index_map[sorted_idx]
                agent_distribution[orig_idx] = optimal_p_prime[sorted_idx]
        else:
            # For randomized strategies, sample to estimate distribution
            num_samples = 1000  # Reduced for speed
            for _ in range(num_samples):
                agent_result = agent_strategy(n, l, t, boxes)
                for i, (val, chosen) in enumerate(agent_result):
                    if chosen == 1:
                        agent_distribution[i] += 1.0
            # Normalize to get probabilities (they should sum to l, not 1)
            for i in range(n):
                agent_distribution[i] /= num_samples
    else:
        # Default: use water fill distribution
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda x: x[1], reverse=True)
        sorted_boxes = [val for _, val in indexed_boxes]
        index_map = [orig_idx for orig_idx, _ in indexed_boxes]
        max_val, optimal_p_prime = water_fill(sorted_boxes, t, l)
        for sorted_idx in range(n):
            orig_idx = index_map[sorted_idx]
            agent_distribution[orig_idx] = optimal_p_prime[sorted_idx]
    
    # Calculate expected values: v_i * p_i for each box
    expected_values = [(boxes[i] * agent_distribution[i], i) for i in range(n)]
    
    # Sort by expected value descending and take top t
    expected_values.sort(reverse=True)
    top_t_indices = [idx for _, idx in expected_values[:t]]
    
    # Return the t boxes with highest expected values
    return [(boxes[i], 1 if i in top_t_indices else 0) for i in range(n)]


if __name__ == "__main__":

    # Scenario 1 (4 boxes, agent picks 1, adversary picks 1)
    scenario1 = setup_scenario([4, 5, 7, 8], l=1, t=1)

    # Scenario 2 (10 boxes, agent picks 3, adversary picks 3)
    scenario2 = setup_scenario([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], l=3, t=3) # values of the boxes can be determined later I just did 1-10 for simplicity

    # Scenario 3 (10 boxes, agent picks 5, adversary picks 2)
    scenario3 = setup_scenario([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], l=5, t=2) # values of the boxes can be determined later I just did 1-10 for simplicity

    # Stores all scenarios
    scenarios = [scenario1, scenario2, scenario3]

    # Stores all agent strategies
    agent_strategies = [pick_randomly_agent, deterministic_agent, greedy_agent, safe_agent, water_fill_agent]

    # Stores all adversary strategies
    adversary_strategies = [pick_randomly_adversary, deterministic_adversary, expected_value_adversary, optimal_byzantine_adversary]

    # If true prints information about every single combination
    verbose = False

    num = 0
    scenario_count = 1

    for scenario in scenarios:
        # Stores average utilites for all strategy combinations for the results summary
        scenario_results = []


        for agent_strategy in agent_strategies:
            for adversary_strategy in adversary_strategies:
                agent_utility, adversary_utility = simulate(scenario, agent_strategy, adversary_strategy, simulations=100)

                # Store the results for this combination
                scenario_results.append({
                    "agent": agent_strategy.__name__,
                    "adversary": adversary_strategy.__name__,
                    "agent_utility": agent_utility,
                    "adversary_utility": adversary_utility
                })

                if verbose:
                    # Counter logic (helps us see how many total combinations have been run)
                    num += 1
                    print(str(num) + ". ")

                    print(f"Scenario: {scenario}, Agent Strategy: {agent_strategy.__name__}, Adversary Strategy: {adversary_strategy.__name__}")
                    print(f"Average Agent Utility: {agent_utility}, Average Adversary Utility: {adversary_utility}\n")
                
        # Summary of results
        best_agent = max(scenario_results, key=lambda x: x["agent_utility"])
        best_adversary = max(scenario_results, key=lambda x: x["adversary_utility"])
        
        # Also find best agent against optimal adversary (water fill should win here)
        optimal_adversary_results = [r for r in scenario_results if r["adversary"] == "optimal_byzantine_adversary"]
        if optimal_adversary_results:
            best_agent_vs_optimal = max(optimal_adversary_results, key=lambda x: x["agent_utility"])

        print(f"--- Scenario {scenario_count}: N = {len(scenario[3])}, T = {scenario[2]} , L = {scenario[1]} ---")
        scenario_count += 1
        print(f"Best Agent Strategy (Overall): {best_agent['agent']} with Average Utility: {best_agent['agent_utility']} and Total Potential Utility: {sum(scenario[3])}")
        if optimal_adversary_results:
            print(f"Best Agent Strategy (vs Optimal Adversary): {best_agent_vs_optimal['agent']} with Average Utility: {best_agent_vs_optimal['agent_utility']} and Total Potential Utility: {sum(scenario[3])}")
        print(f"Best Adversary Strategy: {best_adversary['adversary']} with Average Utility: {best_adversary['adversary_utility']} and Total Potential Utility: {sum(scenario[3])}\n")