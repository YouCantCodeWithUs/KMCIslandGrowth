#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
from matplotlib import pyplot as plt

# Constants
E_a = 2.0972
E_s = 2.0972
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
nbins_x = int(L/bin_size)
nbins_y = int(D/bin_size)

def InitSubstrate(w, h, r_s):
	'''
	Initialize substrate atom positions in a hexagonal grid below y = 0
	
	w:   Int   - number of atoms in a horizontal row
	h:   Int   - number of rows
	r_s: Float - spacing between atoms
	
	returns: np.array(np.array[x, y]) - list of substrate atom positions
	'''
	R = []
	roffset = np.array([r_s/2, 0])
	for i in range(h):
		y = -i*r_s*sqrt3/2
		for x in range(w):
			Ri = np.array([x*r_s, y])
			if i%2 == 1:
				Ri += roffset
			R.append(PutInBox(Ri))
	return np.array(R)

def PutInBox(Ri):
	'''
	Implement periodic boundary conditions along the x-axis.
	
	Ri: np.array[x, y] - position of a single atom
	L:  Float          - global variable indicating the width of the periodic cell
	
	returns: np.array[x, y] - correct position using PBC
	'''
	x = Ri[0] % L
	if abs(x) > L/2:
		x += L if L < 0 else -L
	return [x, Ri[1]]

def Displacement(Ri, Rj):
	'''
	Least-distance spacial displacement between two atoms.
	
	Ri: np.array[x, y] - position of first atom
	Rj: np.array[x, y] - position of second atom
	
	returns: np.array[x, y] - vector pointing from Ri to Rj
	'''
	return PutInBox(Rj-Ri)

def Distance(Ri, Rj):
	'''
	Least-distance between two atoms.
	
	Ri: np.array[x, y] - position of first atom
	Rj: np.array[x, y] - position of second atom
	
	returns: Float - distance between Ri and Rj
	'''
	d = Displacement(Ri, Rj)
	return np.sqrt(np.dot(d, d))

def BinIndex(Ri):
	'''
	Position of an atom in 'bin-space'.
	
	Ri:       np.array[x, y] - position of atom
	L:        Float          - global variable indicating width of the periodic cell
	bin_size: Float          - global variable indicating width of a bin
	Dmin:     Float          - global variable indicating the lowest substrate atom y-position
	
	returns: (Int, Int) - indices of the atom in 'bin-space'
	'''
	nx = int((Ri[0] + L/2)/bin_size)
	ny = int((Ri[1] - Dmin)/bin_size)
	return (nx, ny)

def PutInBins(R):
	'''
	Which atoms go in which bins?
	
	R: np.array(np.array[x, y]) - list of atom positions
	
	returns: [[[np.array[x, y]]]] - bins of atom positions ordered column-first ([x][y])
	'''
	bins = []
	for Ri in R:
		(nx, ny) = BinIndex(Ri)
		while len(bins) <= nx:
			bins.append([])
		while len(bins[nx]) <= ny:
			bins[nx].append([])
		bins[nx][ny].append(Ri)
	return bins

def NearbyBins(Ri):
	'''
	Which bins are adjacent to this atom?
	
	Ri:      np.array[x, y] - position of atom
	nbins_x: Int            - global variable indicating number of bins along x-axis
	nbins_y: Int            - global variable indicating number of bins along y-axis
	
	returns: ([Int], [Int]) - indices of nearby bins separated by coordinate ([all-xs], [all-ys])
	'''
	(nx, ny) = BinIndex(Ri)
	nearby_x = [nx-1, nx, nx+1]
	nearby_y = [ny-1, ny, ny+1]
	for i in range(3):
		if nearby_x[i] < 0:
			nearby_x[i] += nbins_x
		if nearby_x[i] >= nbins_x:
			nearby_x[i] -= nbins_x
		if nearby_y[i] < 0:
			nearby_y[i] += nbins_y
		if nearby_y[i] >= nbins_y:
			nearby_y[i] -= nbins_y
	return (nearby_x, nearby_y)

def PairwisePotential(Ri, Rj, E_ij, r_ij):
	'''
	Lennard-Jones potential between two atoms.
	
	Ri:   np.array[x, y] - position of first atom
	Rj:   np.array[x, y] - position of second atom
	E_ij: Float          - binding energy between Ri and Rj
	r_ij: Float          - equilibrium bond length between Ri and Rj
	
	returns: Float - potential energy of Ri due to Rj (will return 0 if Ri == Rj)
	'''
	r = Distance(Ri, Rj)
	if r == 0:
		return 0
	return 4*E_ij*((r_ij/r)**12 - (r_ij/r)**6)

def AdatomSurfaceEnergy(adatom, substrate_bins, E_as, r_as):
	'''
	Lennard-Jones potential between an adatom and the substrate atoms.
	
	adatom:         np.array[x, y]       - position of adatom
	substrate_bins: [[[np.array[x, y]]]] - binned substrate positions
	E_as:           Float                - binding energy between adatoms and substrate atoms
	r_as:           Float                - equilibrium bond length between adatoms and substrate atoms
	
	returns: Float - potential energy of adatom due to substrate atoms
	'''
	(nearby_x, nearby_y) = NearbyBins(adatom)
	nearby_substrate = []
	for x in nearby_x:
		for y in nearby_y:
			try:
				nearby_substrate += substrate_bins[x][y]
			except IndexError:
				pass # Bins with no atoms are not returned from PutInBins()
	return sum([PairwisePotential(adatom, Rs, E_as, r_as) for Rs in nearby_substrate])

def AdatomAdatomEnergy(i, adatoms, E_a, r_a):
	'''
	Lennard-Jones potential between an adatom and other adatoms.
	
	i:       Int                      - index of adatom
	adatoms: np.array(np.array[x, y]) - position of adatom
	E_a:     Float                    - binding energy between adatoms and substrate atoms
	r_a:     Float                    - equilibrium bond length between adatoms and substrate atoms
	
	returns: Float - potential energy of an adatom due to other adatoms
	'''
	Ri = adatoms[i]
	# faster for small systems, slower for big systems
	if len(adatoms) < 1000:
		return sum([PairwisePotential(Ri, Rj, E_as, r_as) for Rj in adatoms])
	(nearby_x, nearby_y) = NearbyBins(Ri)
	adatom_bins = PutInBins(adatoms)
	nearby_adatoms = []
	for x in nearby_x:
		for y in nearby_y:
			try:
				nearby_adatoms += adatom_bins[x][y]
			except IndexError:
				pass # Bins with no atoms are not returned from PutInBins()
	return sum([PairwisePotential(Ri, Rj, E_as, r_as) for Rj in nearby_adatoms])

def PlotSubstrate(substrate):
	for a in substrate:
		plt.scatter(a[0], a[1], color='blue')
	for x in range(nbins_x + 2):
		plt.plot([x*bin_size - L/2, x*bin_size - L/2], [Dmin, nbins_y*bin_size + Dmin], color='red')
	for y in range(nbins_y + 1):
		plt.plot([-L/2, (nbins_x + 1)*bin_size - L/2], [y*bin_size + Dmin, y*bin_size + Dmin], color='red')
	plt.show()

substrate = InitSubstrate(W, Hs, r_s)
substrate_bins = PutInBins(substrate)

# minimum energy position
# xs = np.linspace(0, L, 500)
# Es = [AdatomSurfaceEnergy(np.array([x, r_as]), substrate_bins, E_as, r_as) for x in xs]
# xmin = xs[Es.index(min(Es))]
# xs = np.linspace(xmin-0.1, xmin+0.1, 500)
# Es = [AdatomSurfaceEnergy(np.array([x, r_as]), substrate_bins, E_as, r_as) for x in xs]
# xmin = xs[Es.index(min(Es))]
# ys = np.linspace(r_as/2, r_as*2, 500)
# Es = [AdatomSurfaceEnergy(np.array([xmin, y]), substrate_bins, E_as, r_as) for y in ys]
# ymin = ys[Es.index(min(Es))]
# ys = np.linspace(ymin-0.1, ymin+0.1, 500)
# Es = [AdatomSurfaceEnergy(np.array([xmin, y]), substrate_bins, E_as, r_as) for y in ys]
# ymin = ys[Es.index(min(Es))]
# print xmin/r_as, ymin/r_as
# print AdatomSurfaceEnergy(np.array([xmin, ymin]), substrate_bins, E_as, r_as)
# plt.plot(ys, Es)
# plt.show()