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

simulations = 100
boxes = [4, 5, 7, 8]
simulate(simulations, boxes)



