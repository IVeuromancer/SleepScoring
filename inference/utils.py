import joblib

def load_scalers(paths):
    return [joblib.load(path) for path in paths]
