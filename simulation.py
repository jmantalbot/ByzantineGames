import random

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

def pick_randomly(boxes):
    return boxes[random.randint(0, len(boxes) - 1)]

def simulate(simulations, boxes):
    max_utility = 0
    total_utility = 0
    for box in boxes:
        max_utility += box
    max_utility *= simulations
    # This is how maso what ny times the adversary successfully zeros out the value.
    times_foiled = 0
    lost_utility = 0
    for i in range(simulations):
        agent_strategy = pick_randomly(boxes)
        adversary_strategy = pick_randomly(boxes)
        if agent_strategy == adversary_strategy:
            times_foiled += 1
            lost_utility += agent_strategy
        else:
            total_utility += agent_strategy
    print(f"The agent had a total of {total_utility} utility out of a maximum of {max_utility}")
    print(f"The adversary foiled the agent {times_foiled} times resulting in a loss of {lost_utility}.")


if __name__ == "__main__":

    # Scenario 1 (4 boxes, agent picks 1, adversary picks 1)
    scenario1 = setup_scenario([4, 5, 7, 8], l=1, t=1)

    # Scenario 2 (10 boces, agent picks 3, adversary picks 3)
    scenario2 = setup_scenario([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], l=3, t=3) # values of the boxes can be determined later I just did 1-10 for simplicity

    # Scenario 3 (10 boxes, agent picks 5, adversary picks 2)
    scenario3 = setup_scenario([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], l=5, t=2) # values of the boxes can be determined later I just did 1-10 for simplicity

    # TODO: Run the simulation 100x for every combination of scenario, agent strategy, and adversary strategy. Find the average of both utilities for each combination.

    # TODO: Print the results in a nice table format. (Determine which strategy was the best for each scenario)

    # This code is used for the current simulate implementation (bare bones)
    simulations = 100
    boxes = [4, 5, 7, 8]
    simulate(simulations, boxes)



