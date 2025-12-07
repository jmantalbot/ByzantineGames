import math
'''
Computes the maximum worst-case EV and the optimal pseudo-distribution p' (p-prime)

Arguments: 
  values: list of reported skill values, assumed to be sorted
  t: Number of byzantine agents
  l: number of agents to select (l > 0)

Returns:
  (max_value, optimal_p_prime)
'''
def water_fill(values, t, l):
  n = len(values)
  # This is an edge case. Basically the EV is trivially 0
  if t >= l and l == n:
    return 0.0, [0.0] * n
  # Case 1: if selecting 1 agent
  if l == 1:
    max_val, optimal_p_prime = solve_l1(values, t, n)
  # Case 2: if selecting 2 or more agents
  else:
    max_val, optimal_p_prime = solve_l(values, t, l, n)
  return max_val, optimal_p_prime



'''
1. The case where $ell = 1$
Finds the optimal value using the solution for l=1

Arguments:
  v: list of reported skill values, assumed to be sorted
  t: number of byzantine agents
  n: number of boxes
'''
def solve_l1(v, t, n):
  max_val = 0.0
  optimal_i = t
  # Selecting a prefix of len i >= t + 1
  # For each prefix length i, calculate sum_{j=0}^{i-1} 1/v_j (0-indexed)
  for i in range(t + 1, n + 1):
    # Calculate sum for current prefix len i: sum of 1/v_j for j=0 to i-1
    inverse_v_sum = sum(1.0 / v[j] for j in range(i))
    # calculate value for prefix len i -> value(p^i) = (i-t) / (sum_{j=0}^{i-1} 1/v_j)
    current_val = (i - t) / inverse_v_sum if inverse_v_sum > 0 else 0.0
    if current_val > max_val:
      max_val = current_val
      optimal_i = i
  # construct optimal pseudo-distribution p'
  if max_val > 0:
    # T* = max_value, p_j = T* / v_j
    p_prime = [0.0] * n
    for j in range(optimal_i):
      p_prime[j] = max_val / v[j]
    # We create a normalizing factor C
    C = sum(1.0 / v[j] for j in range(optimal_i))
    final_p = [(1.0 / v[j]) / C if j < optimal_i else 0.0 for j in range(n)]
    return max_val, final_p
  else:
    return 0.0, [0.0] * n




'''
The case where $ell > 1$
Constructs the maximal E-nice pseudo-distribution and calculates its water usage

Arguments:
  v: sorted value boxes
  n: number of boxes
  E: constant minimum expected value assigned to critical boxes
  l: number of boxes to select

Returns:
  total water used, pseudo-distribution, saturation_index_i
'''

def get_p_prime_for_E(v, n, E, l):
  p_prime = [0.0] * n
  total_water = 0.0
  i_saturation = n #last non-empty box
  
  for k in range(n):
    v_k = v[k]
    # desired prob to reach level E
    desired_p_k = E / v_k
    # max prob is 1
    p_k = min(1.0, desired_p_k)
    
    # apply l budged contraint
    if total_water + p_k > l:
      p_k = l - total_water # use remaining water to partial fill container
      i_saturation = k - 1 # container k is partially filled
      p_prime[k] = p_k
      total_water = l
      break
    p_prime[k] = p_k
    total_water += p_k
    i_saturation = k # container k is now saturated
  return total_water, p_prime, i_saturation


'''
The Case where $ell > 1$
Finds the optimal value by simulating the continuous decreease of E and finding breakpoints

Arguments:
  v: list of sorted boxes
  t: number of byzantine boxes
  l: number to choose
  n: number of boxes
'''
def solve_l(v,t,l,n):
  # Determine E_max
  # (i) E <= v_{t+1}
  v_t_plus_1 = v[t] if t<n else 0.0
  # (ii) E <= l / sum_{k=1}^{t+1} (1/v_k)
  sum_inv_v_t_plus_1 = sum(1.0 / v[k] for k in range(t+1))
  E_max_l_constraint = l / sum_inv_v_t_plus_1 if sum_inv_v_t_plus_1 > 0 else float('inf')
  E_max = min(v_t_plus_1, E_max_l_constraint)
  if E_max <= 0:
    return 0.0, [0.0] * n
  max_value = 0.0
  optimal_p_prime = [0.0] * n

  # The optimal value must be achieved at a breakpoint of E
  # Breakpoints occur when a container becomes full or is emptied
  # Breakpoint Iteration, simulating decreasing E
  # Start E at E_max
  breakpoints = {E_max, 0.0} # candidates to check
  # Additional breakpoints happen when E = v_k for any k > t + 1
  # wwhere container k transitions from partial fill to full or from full to requiring E/v_k < 1
  for k in range(t+1, n):
    if v[k] <= E_max:
      breakpoints.add(v[k])
  sorted_breakpoints = sorted(list(breakpoints), reverse=True)
# Check each breakpoint
  for E in sorted_breakpoints:
  # calculate maximal E-nice pseudo-distribution p'
    total_water, p_prime, i_saturation = get_p_prime_for_E(v,n,E,l)
  # the EV is the water level E times the number of surviving boxes (i_saturation - t + 1) plus the value of the partially ffilled box (i_saturation + 1) if it exists
  # Value(p') = min_{B in [n]^{t}} sum_{j=1}^{n} v_j * p'_j
  # THE ADVERSARY CHOOSES B TO NULLIFY THE T BOXES WITH THE LARGEST EV
  # Since p' is E-nice, E is the largest EV for j <= t + 1
  # THE ADVERSARY WILL NULLIFY THE T BOXES WITH EXPECTED VALUE E
    current_value = 0.0
    value_if_t_highest_killed = sum(v[j] * p_prime[j] for j in range(t,n))
  # We need to find the worst-case B. The worse case B nullifies the t boxes with the largest EV.
  # the value is determined by the boxes not killed by the adversary.
    current_value = sum(v[j] * p_prime[j] for j in range(t,n))
    if current_value > max_value:
      max_value = current_value
      optimal_p_prime = p_prime
  return max_value, optimal_p_prime

