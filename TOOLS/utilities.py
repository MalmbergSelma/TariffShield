import pickle

def save_result(res, name):
    """Saves the result res in a pickle object name.pickle."""
    try:
        with open(name + ".pickle", "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print('Error during pickling object:', ex) 


def load_result(filename):
    """Load results stocked in filename."""
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object:", ex)