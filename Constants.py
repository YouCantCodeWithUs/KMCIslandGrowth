#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np

# Constants
E_a = 2.0972
E_s = 2.0972
# r_a and r_s have to be scaled for the orientation being used?
r_a = 2.55266
r_s = 2.55266
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