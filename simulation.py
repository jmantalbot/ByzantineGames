import random


def pick_randomly(boxes):
    return boxes[random.randint(0, len(boxes) - 1)]

def simulate(simulations, boxes):
    max_utility = 0
    total_utility = 0
    for box in boxes:
        max_utility += box
    max_utility *= simulations
    # This is how many times the adversary successfully zeros out the value.
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



