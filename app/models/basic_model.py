# app/models/basic_model.py

import numpy as np

class BasicModel:
    def predict(self, data):
        """ Dummy trading signal: random choice. """
        return np.random.choice(["buy", "sell", "hold"])
