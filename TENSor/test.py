import numpy as np
import tensorflow as tf
import pickle


class Animals:
    def breath(self):
        print('breathing')
    def move(self):
        print('moving')


class Mammals(Animals):
    def breastfeed(self):
        print('feeding young')


class Cats(Mammals):
    def __init__(self, spots):
        self.spots = spots
    def catch_mouse(self):
        print('catch mouse')


game_data = {
    'position': 'E2 N3',
    'health': 100,
    'pocket': ['keys', 'knife']
}

save_file = open('data.dat', 'wb')
pickle.dump(game_data, save_file)
save_file.close()

load_file = open('data.dat', 'rb')
xx = pickle.load(load_file)
load_file.close()
print(xx)