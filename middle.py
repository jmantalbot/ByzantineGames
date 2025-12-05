def optimal_randomized_agent(n, l, t, boxes, optimal_p_prime):
  p_prime_array = np.arrray(optimal_p_prime)
  chosen_boxes_indices = []
  # Remaining selections
  R_rem = l
  # remaining probability mass available
  W_rem = np.sum(p_prime_array)
  for i in range(n):
    p_i = p_prime_array[i]
    # Calculate adjusted probability to select box i
    # P_i ensures marginals sum up correctly
    # P(select i) = min(p_i * R_rem / W_rem, (R_rem - W_rem + p_i) / (R_rem - W_rem + p_i))
    # A simple, commonly used coupling probability P(select i | marginals)
    # to ensure total number of selections is l is:
    if W_rem == 0:
      # should only happen if W_rem and R_rem are both 0 (all boxes processed)
      P_i = 0.0
    else:
      P_i = p_i * R_rem / W_rem
    if random.random() < P_i:
      chosen_boxes_indices.append(i)
      R_rem -= 1
      W_rem -= p_i
    else:
      W_rem -= p_i
    if R_rem == 0:
      break
  return [(boxes[i], 1 if i in chosen_boxes_indices else 0) for i in range(n)]
