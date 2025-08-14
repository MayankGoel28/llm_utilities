import pickle
import os
import inspect

def save_pickle(variable, directory="."):
    # Get the variable name from the caller's frame
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    variable_name = [var_name for var_name, var_val in callers_local_vars if var_val is variable]
    if not variable_name:
        raise ValueError("Could not determine the variable name.")
    file_path = os.path.join(directory, f"{variable_name[0]}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(variable, f)