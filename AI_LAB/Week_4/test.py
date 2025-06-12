def epsilon_closure(states, transitions, visited=None):
    """Compute the epsilon closure for a set of states."""
    if visited is None:
        visited = set()
    closure = set(states)
    for state in states:
        if state not in visited:
            visited.add(state)
            if (state, '') in transitions:  # '' represents epsilon
                next_states = transitions[(state, '')]
                closure.update(epsilon_closure(next_states, transitions, visited))
    return closure

def simulate_epsilon_nfa(input_string):
    """Simulate ε-NFA for the given input string."""
    # Define the ε-NFA
    states = {'q0', 'q1', 'q2'}
    alphabet = {'a', 'b'}
    transitions = {
        ('q0', ''): {'q1'},  # ε-transition
        ('q1', 'a'): {'q2'},
        # Note: No transition defined for q0 with 'a' or 'b', q1 with 'b', or q2 with 'a' or 'b'
    }
    initial_state = 'q0'
    accepting_states = {'q2'}

    # Get initial epsilon closure
    current_states = epsilon_closure({initial_state}, transitions)
    print(f"Initial epsilon closure: {current_states}")

    # Process each symbol in the input string
    for symbol in input_string:
        if symbol not in alphabet:
            print(f"Invalid symbol {symbol} in input")
            return False

        next_states = set()
        for state in current_states:
            if (state, symbol) in transitions:
                next_states.update(transitions[(state, symbol)])

        # Compute epsilon closure for all next states
        current_states = epsilon_closure(next_states, transitions)
        print(f"After reading '{symbol}': {current_states}")

    # Check if any current state is an accepting state
    is_accepted = bool(current_states & accepting_states)
    print(f"String '{input_string}' is {'accepted' if is_accepted else 'not accepted'}")
    return is_accepted

# Test the ε-NFA with input "a"
input_string = "a"
simulate_epsilon_nfa(input_string)