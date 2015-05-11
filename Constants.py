#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
import numpy as np

if len(sys.argv) > 1:
	E_a = float(sys.argv[1])
	E_s = float(sys.argv[2])
	r_a = float(sys.argv[3])
	r_s = float(sys.argv[4])
	W = int(sys.argv[5])
	folder = sys.argv[6]
else:
	# Constants
	E_a = 2.083 # eV, Cu-Cu
	E_s = 2.083 # eV, Cu-Cu
	# r_a and r_s have to be scaled for the orientation being used? also, not the lattice parameters.
	# should be divided by sqrt2
	r_a = 2.7 # angstroms Cu
	r_s = 2.7 # angstroms Cu
	folder = '_Images/'
	
	# Box Parameters
	W = 18

# Constants
E_as = np.sqrt(E_a*E_s)
r_as = (r_a+r_s)/2
sqrt3 = np.sqrt(3)

# Box Parameters
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
beta = 1.0/300/boltzmann/2 # 1/eV
deposition_rate = 1.0