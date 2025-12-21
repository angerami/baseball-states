def is_forbidden_transition(from_token: str, to_token: str) -> bool:
    """Return True if transition violates baseball rules."""
    from_state = parse_state(from_token)
    to_state = parse_state(to_token)
    
    # Special token rules
    if from_token == Vocabulary.START_INNING:
        # Must go to OUT0_* state
        return to_state is None or to_state[0] != 0
    
    if from_token == Vocabulary.END_INNING:
        # END_INNING shouldn't predict anything (sequence over)
        return True
    
    # State-to-state rules
    if from_state is not None and to_state is not None:
        from_outs, _ = from_state
        to_outs, _ = to_state
        # Outs can't decrease
        if to_outs < from_outs:
            return True
    
    # State to END_INNING is always legal
    if from_state is not None and to_token == Vocabulary.END_INNING:
        return False
    
    return False