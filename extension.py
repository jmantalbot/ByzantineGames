import random
import numpy as np
from water_fill import water_fill
from middle import optimal_randomized_agent
from collections import deque


def setup_scenario(box_values, l, t):
    """
    Sets up the scenario with predefined boxes.
    """
    n = len(box_values)
    return (n, l, t, box_values)


def simulate(scenario, agent_strategy, adversary_strategy, simulations, history_size=5):
    """
    Simulates the agent and adversary picking boxes from the given scenario.
    """
    total_agent_utility = 0
    total_adversary_utility = 0

    n, l, t, boxes = scenario
    history = deque([(0, 0)] * history_size, maxlen=history_size)

    is_learning_agent = (agent_strategy.__name__ == 'universal_learning_agent')

    if is_learning_agent:
        print(f"\n|--- STARTING ROUND-BY-ROUND LEARNING ({adversary_strategy.__name__}) ---|")
        print(f"{'Round':<5} | {'Prev Util':<9} | {'Strategy Chosen':<20} | {'Current Util':<15}")
        print("-" * 55)

    for i in range(simulations):
        agent_result = agent_strategy(n, l, t, boxes, history)

        if is_learning_agent:
            agent_chosen, chosen_strategy_name = agent_result
        else:
            agent_chosen = agent_result
            chosen_strategy_name = agent_strategy.__name__

        adversary_chosen = adversary_strategy(n, l, t, boxes)

        current_agent_utility = 0
        current_adversary_utility = 0

        for j in range(len(agent_chosen)):
            if agent_chosen[j][1] == 1:
                if adversary_chosen[j][1] == 1:
                    current_adversary_utility += agent_chosen[j][0]
                else:
                    current_agent_utility += agent_chosen[j][0]

        total_agent_utility += current_agent_utility
        total_adversary_utility += current_adversary_utility

        prev_agent_utility = history[-1][0]
        history.append((current_agent_utility, current_adversary_utility))

        if is_learning_agent and i < 50:
            print(
                f"{i + 1:<5} | {prev_agent_utility:<9.2f} | {chosen_strategy_name:<20} | {current_agent_utility:<15.2f}"
            )
        elif is_learning_agent and i == 50:
            print("[... Rounds 51 to 990 skipped for brevity ...]")

    if is_learning_agent:
        print("-" * 55)

    total_agent_utility /= simulations
    total_adversary_utility /= simulations

    return total_agent_utility, total_adversary_utility


def pick_randomly_agent(n, l, t, boxes, history=None):
    """ Agent strategy: Picks boxes randomly. """
    chosen_boxes = random.sample(range(n), l)
    return [(boxes[i], 1 if i in chosen_boxes else 0) for i in range(n)]


def deterministic_agent(n, l, t, boxes, history=None):
    """ Agent strategy: Picks the top l boxes with the highest values. """
    top_indexes = sorted(range(n), key=lambda i: boxes[i], reverse=True)[:l]
    return [(boxes[i], 1 if i in top_indexes else 0) for i in range(n)]


def greedy_agent(n, l, t, boxes, history=None):
    """ Agent strategy: Picks random boxes from the top t + l boxes. """
    top_indexes = sorted(range(n), key=lambda i: boxes[i], reverse=True)[:(t + l)]
    chosen_boxes = random.sample(top_indexes, l)
    return [(boxes[i], 1 if i in chosen_boxes else 0) for i in range(n)]


def safe_agent(n, l, t, boxes, history=None):
    """ Agent strategy: Picks the l boxes immediately following the top t boxes. """
    top_indexes = sorted(range(n), key=lambda i: boxes[i], reverse=True)[:(t + l)]
    chosen_boxes = top_indexes[t:t + l]
    return [(boxes[i], 1 if i in chosen_boxes else 0) for i in range(n)]

def water_fill_agent(n, l, t, boxes, history=None):
    # CRITICAL: Water fill algorithm requires boxes sorted in descending order (v₁ ≥ v₂ ≥ ... ≥ vₙ)
    # Create list of (original_index, value) pairs and sort by value descending
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)
    
    # Extract sorted values and create mapping from sorted index to original index
    sorted_boxes = [val for _, val in indexed_boxes]
    index_map = [orig_idx for orig_idx, _ in indexed_boxes]  # sorted_idx -> orig_idx
    
    # Call water_fill with sorted boxes
    max_val, optimal_p_prime = water_fill(sorted_boxes, t, l)
    
    # Map probabilities back to original indices
    # optimal_p_prime[i] is the probability for sorted box i, we need it for original box index_map[i]
    original_p_prime = [0.0] * n
    for sorted_idx in range(n):
        orig_idx = index_map[sorted_idx]
        original_p_prime[orig_idx] = optimal_p_prime[sorted_idx]
    
    # Use optimal_randomized_agent with original boxes and remapped probabilities
    return optimal_randomized_agent(n, l, t, boxes, original_p_prime)


# Adversary strategies (simplified/placeholder definitions)

def pick_randomly_adversary(n, l, t, boxes, history=None):
    chosen_boxes = random.sample(range(n), t)
    return [(boxes[i], 1 if i in chosen_boxes else 0) for i in range(n)]


def deterministic_adversary(n, l, t, boxes, history=None):
    top_indexes = sorted(range(n), key=lambda i: boxes[i], reverse=True)[:t]
    return [(boxes[i], 1 if i in top_indexes else 0) for i in range(n)]


def expected_value_adversary(n, l, t, boxes, history=None):
    np_boxes = np.array(boxes, dtype=float)
    if np.sum(np_boxes) == 0:
        probabilities = np.ones(n) / n
    else:
        probabilities = np_boxes / np.sum(np_boxes)
    chosen_boxes = np.random.choice(n, size=t, replace=False, p=probabilities)
    return [(boxes[i], 1 if i in chosen_boxes else 0) for i in range(n)]


def optimal_randomized_agent(n, l, t, boxes, optimal_p_prime):
    # Simple placeholder: random choice of t boxes
    chosen_boxes = random.sample(range(n), t)
    return [(boxes[i], 1 if i in chosen_boxes else 0) for i in range(n)]


def optimal_byzantine_adversary(n, l, t, boxes, history=None):
    max_val, optimal_p_prime = water_fill(boxes, t, l)
    return optimal_randomized_agent(n, l, t, boxes, optimal_p_prime)


# Global state, now including the list of all strategies
agent_state = {
    'current_strategy': None,
    'last_utility': 0,
    'strategy_list': [],
    'strategy_index': 0
}


def universal_learning_agent(n, l, t, boxes, history):
    global agent_state

    # history is (agent_utility, adversary_utility). history[-1][0] is the utility from the previous round.
    current_round_prev_agent_utility = history[-1][0]

    # Switch if utility from the previous round was 0.
    # This ensures the agent is not trapped by the deterministic adversary.
    if current_round_prev_agent_utility == 0 and agent_state['current_strategy'] is not None:
        # Failure detected: Switch to the next strategy in the list
        agent_state['strategy_index'] = (agent_state['strategy_index'] + 1) % len(agent_state['strategy_list'])

    # Get the chosen strategy function
    chosen_strategy_func = agent_state['strategy_list'][agent_state['strategy_index']]

    # Execute the chosen strategy for the current round
    agent_chosen_boxes = chosen_strategy_func(n, l, t, boxes)

    # Update state for the next round
    agent_state['current_strategy'] = chosen_strategy_func
    agent_state['last_utility'] = current_round_prev_agent_utility  # This is still useful for tracking

    return (agent_chosen_boxes, chosen_strategy_func.__name__)


if __name__ == "__main__":

    # Define all strategies the learning agent will cycle through
    LEARNING_AGENT_STRATEGIES = [
        water_fill_agent,
        deterministic_agent,
        greedy_agent,
        pick_randomly_agent,
        safe_agent
    ]
    # Set the list and initial index in the global state
    agent_state['strategy_list'] = LEARNING_AGENT_STRATEGIES
    agent_state['strategy_index'] = 0

    # Scenario 1 (4 boxes, agent picks 1, adversary picks 1)
    scenario1 = setup_scenario([4, 5, 7, 8], l=1, t=1)
    scenarios = [scenario1]

    # Stores all agent strategies for the test suite.
    agent_strategies_to_test = [
        water_fill_agent,
        pick_randomly_agent,
        deterministic_agent,
        greedy_agent,
        safe_agent,
        universal_learning_agent
    ]

    # Stores all adversary strategies
    adversary_strategies = [
        pick_randomly_adversary,
        deterministic_adversary,
        expected_value_adversary,
        optimal_byzantine_adversary
    ]

    verbose = True
    simulations = 1000

    scenario_count = 1

    for scenario in scenarios:
        scenario_results = []
        num = 0

        print(
            f"--- Running Test on Scenario {scenario_count}: N = {len(scenario[3])}, T = {scenario[2]} , L = {scenario[1]} (Simulations: {simulations}) ---")

        for agent_strategy in agent_strategies_to_test:
            for adversary_strategy in adversary_strategies:
                if agent_strategy == universal_learning_agent:
                    # Reset the learning agent's state for each adversary run
                    agent_state['strategy_index'] = 0
                    agent_state['current_strategy'] = agent_state['strategy_list'][0]
                    agent_state['last_utility'] = 0

                agent_utility, adversary_utility = simulate(
                    scenario,
                    agent_strategy,
                    adversary_strategy,
                    simulations=simulations,
                    history_size=1
                )

                # Store the results
                scenario_results.append({
                    "agent": agent_strategy.__name__,
                    "adversary": adversary_strategy.__name__,
                    "agent_utility": agent_utility,
                    "adversary_utility": adversary_utility
                })

                if verbose:
                    num += 1
                    print(
                        f"{num:02}. Agent: {agent_strategy.__name__:<25} vs. Adversary: {adversary_strategy.__name__:<25} | Agent Avg Util: {agent_utility:.2f} | Adv Avg Util: {adversary_utility:.2f}")

        # Filter and summarize results
        agent_performance = {}
        for result in scenario_results:
            name = result['agent']
            if name not in agent_performance or result['agent_utility'] > agent_performance[name]['agent_utility']:
                agent_performance[name] = result

        best_agent_overall = max(agent_performance.values(), key=lambda x: x['agent_utility'])

        print(f"\n--- Scenario {scenario_count} Summary ---")
        print(f"Total Potential Utility: {sum(scenario[3])} | Max Single Pick: 8")
        print(
            f"The best overall agent strategy was **{best_agent_overall['agent']}** with Avg Utility of **{best_agent_overall['agent_utility']:.2f}** (achieved vs. {best_agent_overall['adversary']})")
        print("---")

        scenario_count += 1