import pickle
import pandas as pd

pickle_file = open("recipe_corpus.pkl", "rb")
recipe_data = pickle.load(pickle_file)