# Meeting 1 
11/19/2025

All members attended this meeting.

## Project Code Planning

All code will be written in terms of functions and calling those functions, no classes needed... yet.

Discussion over how to organize simulation and implement scenario 1.

Agent strategies: Deterministic, Greedy, Random, Safe, Water_Fill

Adversary strategies: Random, Deterministic, Expected Value.

### Scenarios

Scenario 1: n = 4, t = 1, l = 1
Scenario 2: n = 10, t = 3, l = 3
Scenario 3: n = 10, t = 2, l = 5

Values for boxes have been established for Scenario 1. Start there and then see about the best values for future scenarios.

Nathan can get a framework up by tonight for everyone to look at.

# Meeting 2 11/21

Simulation function\
tuple for different scenarios (n, t, l)\
different functions for strategies (both agent and adversary) give n, t, l\
  return boxes\
function to apply the strategy \
  return values of boxes

Compare agent's choices to updated box list\
Display utility for Agent and utility lost from Adversary\
Learning algorithm (Look at last five outcomes, choose strategy differently).\


### Extensions:
We are proving the math of this mixed strategy, 
proof of concept that the agent and adversary could learn to apply different strategies based on past experiences. Adversary never zeros out top values, so picking top values is actually better than water fill.

# Discord Checkpoint 1 11/21
Natalia modified the main function, created the setup scenario function. modified the simulate function, and modified the pick randomly function.

# Discord Checkpoint 2 11/25
Natalia posted a list of what needs to be done and what has been done. Josh agreed to figure out the water fill strategy, and Nathan agreed on the research component. 

# Discord Checkpoint 3 12/1
Check in to rediscuss dates for completion and plans to finish by the weekend.

# Discord Checkpoint 4 12/4
Plans made to finish by Sunday. Josh is done with the water fill, just putting it in the code. Nathan plans to complete the learning logic over the next few days, and reports can be written.
