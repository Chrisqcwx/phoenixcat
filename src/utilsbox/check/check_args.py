def only_one_given(*args):
    if sum(1 for arg in args if arg is not None) == 1:
        return True
    else:
        return False