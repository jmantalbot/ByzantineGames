import random

import numpy as np

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


def simulate(scenario, agent_strategy, adversary_strategy, simulations):
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

    # Run each simulation
    for i in range(simulations):
        # Agent picks boxes based on its strategy
        agent_chosen = agent_strategy(n, l, t, boxes)
        # Adversary picks boxes based on its strategy
        adversary_chosen = adversary_strategy(n, l, t, boxes)

        # Calculate utilities for every box
        for j in range(len(agent_chosen)):
            # if the agent chose that box
            if agent_chosen[j][1] == 1:
                # check if the adversary also chose that box
                if adversary_chosen[j][1] == 1:
                    # adversary foiled the agent
                    total_adversary_utility += agent_chosen[j][0]
                else:
                    # agent gets the box value as utility because he was not foiled
                    total_agent_utility += agent_chosen[j][0]

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
    agent_strategies = [pick_randomly_agent, deterministic_agent, greedy_agent, safe_agent]  # TODO: Add more agent strategies here

    # Stores all adversary strategies
    adversary_strategies = [pick_randomly_adversary, deterministic_adversary, expected_value_adversary]  # TODO: Add more adversary strategies here

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

        print(f"--- Scenario {scenario_count}: N = {len(scenario[3])}, T = {scenario[2]} , L = {scenario[1]} ---")
        scenario_count += 1
        print(f"Best Agent Strategy: {best_agent['agent']} with Average Utility: {best_agent['agent_utility']} and Total Potential Utility: {sum(scenario[3])}")
        print(f"Best Adversary Strategy: {best_adversary['adversary']} with Average Utility: {best_adversary['adversary_utility']} and Total Potential Utility: {sum(scenario[3])}\n")


