import random
import numpy as np
from water_fill import water_fill
from middle import optimal_randomized_agent
from collections import deque
from simulation import pick_randomly_agent, deterministic_agent, greedy_agent, safe_agent, water_fill_agent
from simulation import pick_randomly_adversary, deterministic_adversary, expected_value_adversary, optimal_byzantine_adversary

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


def simulate(scenario, agent_strategy, adversary_strategy, simulations, history_size=5):
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
    history = deque([(0, 0)] * history_size, maxlen=history_size)

    # NEW: Flag to indicate we need to document round-by-round decisions
    is_learning_agent = (agent_strategy.__name__ == 'simple_learning_agent')

    # NEW: Print headers if this is the learning agent run
    if is_learning_agent:
        print(f"\n|--- STARTING ROUND-BY-ROUND LEARNING ({adversary_strategy.__name__}) ---|")
        print(f"{'Round':<5} | {'Prev Util':<9} | {'Strategy Chosen':<20} | {'Current Util':<15}")
        print("-" * 55)

    for i in range(simulations):
        # Call agent_strategy (returns (boxes, strategy_name) if learning, else just boxes)
        agent_result = agent_strategy(n, l, t, boxes, history)

        if is_learning_agent:
            agent_chosen, chosen_strategy_name = agent_result
        else:
            agent_chosen = agent_result
            chosen_strategy_name = agent_strategy.__name__  # Use original name

        # Adversary picks boxes based on its strategy
        adversary_chosen = adversary_strategy(n, l, t, boxes)

        current_agent_utility = 0
        current_adversary_utility = 0

        # Calculate utilities for every box
        for j in range(len(agent_chosen)):
            if agent_chosen[j][1] == 1:
                if adversary_chosen[j][1] == 1:
                    current_adversary_utility += agent_chosen[j][0]
                else:
                    current_agent_utility += agent_chosen[j][0]

        # Update totals and history
        total_agent_utility += current_agent_utility
        total_adversary_utility += current_adversary_utility

        # We need the utility *from the previous round* for history[-1][0],
        # but the current utility for documentation.
        prev_agent_utility = history[-1][0]
        history.append((current_agent_utility, current_adversary_utility))

        # NEW: Document the decision and outcome for the learning agent
        if is_learning_agent and i < 10:  # Only print the first 10 rounds for brevity
            print(
                f"{i + 1:<5} | {prev_agent_utility:<9.2f} | {chosen_strategy_name:<20} | {current_agent_utility:<15.2f}"
            )
        elif is_learning_agent and i == 10:
            print("[... Rounds 11 to 990 skipped for brevity ...]")

    #Print final line for learning agent run
    if is_learning_agent:
        print("-" * 55)

    # Compute average utilities
    total_agent_utility /= simulations
    total_adversary_utility /= simulations

    return total_agent_utility, total_adversary_utility

agent_state = {
    'current_strategy': None,  # Will be set to deterministic_agent in main
    'last_utility': 0
}

def simple_learning_agent(n, l, t, boxes, history):
    global agent_state

    """
    Agent strategy: Switches between deterministic_agent and greedy_agent
    based on the utility from the last round.

    NOTE: The 'greedy_agent' must be correctly defined elsewhere in the file.
    """


    # Check the last utility (the last item in the history deque)
    # The first few times the history will contain only zeros, which is fine.
    last_agent_utility = history[-1][0]

    # Switching Logic: If the agent was foiled (utility was 0), switch strategy.
    if last_agent_utility == 0 and agent_state['last_utility'] != 0:
        # Strategy failed (was foiled). Switch!
        if agent_state['current_strategy'] == deterministic_agent:
            chosen_strategy_func = greedy_agent
        else:
            chosen_strategy_func = deterministic_agent

    else:
        # Keep the current strategy (or choose deterministic if first run)
        if agent_state['current_strategy'] is None:
            chosen_strategy_func = deterministic_agent
        else:
            chosen_strategy_func = agent_state['current_strategy']

    # Execute the chosen strategy for the current round
    agent_chosen_boxes = chosen_strategy_func(n, l, t, boxes)

    agent_state['current_strategy'] = chosen_strategy_func

    # Store utility from previous round (history[-1][0])

    agent_state['last_utility'] = last_agent_utility

    return (agent_chosen_boxes, chosen_strategy_func.__name__)

if __name__ == "__main__":

    # Scenario 1 (4 boxes, agent picks 1, adversary picks 1)
    # Boxes: [4, 5, 7, 8]. N=4, L=1, T=1.
    scenario1 = setup_scenario([4, 5, 7, 8], l=1, t=1)
    # Just testing scenario 1 for now
    scenarios = [scenario1]

    # Stores all agent strategies for the test suite.
    agent_strategies_to_test = [
        pick_randomly_agent,
        deterministic_agent,
        greedy_agent,
        simple_learning_agent
    ]

    # Stores all adversary strategies
    adversary_strategies = [
        pick_randomly_adversary,
        deterministic_adversary,
        expected_value_adversary,
        optimal_byzantine_adversary
    ]

    # If true prints information about every single combination
    verbose = False
    simulations = 1000  # Increased for better statistical stability

    # Simulation Loop

    scenario_count = 1

    for scenario in scenarios:
        # Stores average utilites for all strategy combinations for the results summary
        scenario_results = []
        num = 0

        print(
            f"--- Running Test on Scenario {scenario_count}: N = {len(scenario[3])}, T = {scenario[2]} , L = {scenario[1]} (Simulations: {simulations}) ---")

        for agent_strategy in agent_strategies_to_test:
            for adversary_strategy in adversary_strategies:
                if agent_strategy == simple_learning_agent:
                    agent_state['current_strategy'] = deterministic_agent  # Starting strategy
                    agent_state['last_utility'] = 0

                agent_utility, adversary_utility = simulate(
                    scenario,
                    agent_strategy,
                    adversary_strategy,
                    simulations=simulations,
                    history_size=1  # Simple learning only needs the last outcome
                )

                # Store the results
                scenario_results.append({
                    "agent": agent_strategy.__name__,
                    "adversary": adversary_strategy.__name__,
                    "agent_utility": agent_utility,
                    "adversary_utility": adversary_utility
                })

                if verbose or agent_strategy == simple_learning_agent:
                    num += 1
                    print(
                        f"{num:02}. Agent: {agent_strategy.__name__:<25} vs. Adversary: {adversary_strategy.__name__:<25} | Agent Avg Util: {agent_utility:.2f} | Adv Avg Util: {adversary_utility:.2f}")

        best_agent_run = max(scenario_results, key=lambda x: x["agent_utility"])

        # Filter results to find the best performing strategy overall for the scenario
        # Group by agent type
        agent_performance = {}
        for result in scenario_results:
            name = result['agent']
            if name not in agent_performance or result['agent_utility'] > agent_performance[name]['agent_utility']:
                agent_performance[name] = result

        # Find the best agent strategy overall
        best_agent_overall = max(agent_performance.values(), key=lambda x: x['agent_utility'])

        print(f"\n--- Scenario {scenario_count} Summary ---")
        print(f"Total Potential Utility: {sum(scenario[3])} | Max Single Pick: 8")
        print(
            f"The best overall agent strategy was **{best_agent_overall['agent']}** with Avg Utility of **{best_agent_overall['agent_utility']:.2f}** (achieved vs. {best_agent_overall['adversary']})")
        print("---")

        scenario_count += 1
