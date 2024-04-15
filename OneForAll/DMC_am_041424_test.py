import numpy as np
import DMC_am_041424 as code

ptcl1 = 0
ptcl2 = 1
ptcl3 = 2

walkers = code.Walkers({'ptcl1': 1, 'ptcl2': 1, 'ptcl3': 1})

walkers.set_arr(np.array([[[0, 0, 0], [0, 0, 1], [0, 1, 0]], [[2, 1, 1], [4, 3, 4], [6, 5, 5]], [[3, 4, 2], [5, 4, 2], [4, 5, 4]]]), isUpdate=False)

bond1 = code.bond_vec(walkers, ptcl1, ptcl2)
bond2 = code.bond_vec(walkers, ptcl1, ptcl3)

print(f'bond1:\n{bond1}')
print(f'bond2:\n{bond2}')

bond_ang = code.bond_angle(bond1, bond2)

print(bond_ang)