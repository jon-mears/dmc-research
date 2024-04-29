import DMC_am_042824 as code

import numpy as np

particles2mass = {'C': 12.0000, 'O': 15.995, 'Au': 80, 'Ag': 90}
molecules2counts = {'CO': 1, 'Au': 1}

w = code.Walkers(particles2mass, molecules2counts)

arr = np.array([[[[0, 0, 0], [3, 0, 0]], [[0, 0, 0], [5, 0, 0]]], [[[0, 2, 0], [0, 4, 0]], [[0, 3, 0], [0, 4, 0]]]])

w.set_arr(arr, False)


vecs = code.vectors(w, 'CO 1')

print(vecs[1, :, w.ptclidx['CO C'], :, :])
# print(bonds[:, w.mlclidx['CO'], w.mlclidx['CO'], w.ptclidx['CO O'], :])