#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np

# Constants
E_a = 1.688 # eV, Ag-Ag
E_s = 2.083 # eV, Cu-Cu
# r_a and r_s have to be scaled for the orientation being used? also, not the lattice parameters.
# should be divided by sqrt2
# also don't use this for sigma. use r_m formula on wiki.
r_a = 3.2 # angstroms Ag
r_s = 2.7 # angstroms Cu
E_as = np.sqrt(E_a*E_s)
r_as = (r_a+r_s)/2
sqrt3 = np.sqrt(3)

# Box Parameters
W = 18
Hs = 10
Ha = 10
L = W*r_s
Dmin = -(Hs)*r_s*sqrt3/2
Dmax = Ha*r_s
D = Dmax-Dmin
bin_size = 3*r_s
nbins_x = int(np.ceil(L/bin_size))
nbins_y = int(np.ceil(D/bin_size))

# Environmental Parameters
boltzmann = 8.617e-5 #eV/K
beta = 1.0/300/boltzmann/6 # 1/eV
deposition_rate = 1.0