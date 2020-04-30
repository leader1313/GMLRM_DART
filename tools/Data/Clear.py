import os

def clear():
    action_dir = "data/action"
    action_list = [ f for f in os.listdir(action_dir) if f.endswith(".xlsx") ]
    state_dir = "data/state"
    state_list = [ f for f in os.listdir(state_dir) if f.endswith(".xlsx") ]
    for f in action_list:
        os.remove(os.path.join(action_dir, f))
    for f in state_list:
        os.remove(os.path.join(state_dir, f))
