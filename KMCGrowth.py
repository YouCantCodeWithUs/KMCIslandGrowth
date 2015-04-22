#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys, random
import numpy as np
from matplotlib import pyplot as plt

'''
Constants to Play With:

Taking Cu(111) as substrate.
Cu has a lattice constant of 3.61A.
Nearest neighbor distance for FCC = a*sqrt(2)/2 = 2.55266A.
Cu has a bond energy of __eV

E_a = adatom-adatom binding energy (eV)
E_s = substrate-substrate binding energy (eV)
E_as = adatom-substrate binding energy
r_a = equilibrium adatom bond length/nearest neighbor distance (Angstroms)
r_s = equilibrium substrate bond length/nearest neighbor distance (Angstroms)
r_as = adatom-substrate bond length/nearest neighbor distance

First config: Substrate and Film have same bond energies and lattice constants
Second config:
Third config:
'''

E_a = 2.0972
E_s = 2.0972
r_a = 2.55266
r_s = 2.55266
E_as = np.sqrt(E_a*E_s)
r_as = (r_a+r_s)/2

#atoms per layer
w = 18
#box length
L = (w+1)*r_s
#substrate layers
l = 11
#substrate depth
D = l*r_s*np.sqrt(3)*0.5
#maximum film layers
F = 10.0
#maximum film height
H = F*r_a*np.sqrt(3)*0.5
#grid size
bin_size = 3*r_s
nbins_x = int(L/bin_size)
nbins_y = int((D+H)/bin_size)

def InitSubstrate():
	positions = []
	roffset = r_s/2
	for i in range(0, l):
		z = r_s*i*np.sqrt(3)*0.5
		for x in range(0, w):
			if(i%2 == 0):
				#layer is even
				positions.append([x*r_s,z,0])
			else:
				#layer is odd
				positions.append([(x*r_s + roffset), z,0])
	return positions

def PutInBox(Ri):
    # Ri[0] is the x coordinate of the ith particle.
    # Similarly, Ri[1] (or Ri[2]) is the y (or z) coordinate.
    # Using periodic boundary conditions, redefine Ri so that it is in a box
    # of side length L centered about about the origin.
    # <-- Modify Ri. -->
	Ri[0] += L/2.0
	Ri[1] += L/2.0
	Ri[2] += L/2.0
	Ri[0] = Ri[0]%L - L/2.0
	Ri[1] = Ri[1]%L - L/2.0
	Ri[2] = Ri[2]%L - L/2.0
	return

def BinIndices(Ri):
	nx = int(Ri[0]/bin_size)
	ny = int(Ri[1]/bin_size)
	return (nx, ny)

def PutInBins(R):
	#Sorts atoms into bins of size 3*r_a to assist with truncation of interactions.
	bins = []
	for Ri in R:
		(nx, ny) = BinIndices(Ri)
		while len(bins) <= nx:
			bins.append([])
		while len(bins[nx]) <= ny:
			bins[nx].append([])
		bins[nx][ny].append(Ri)
	return bins


def Distance(Ri,Rj):
    # Again, Ri is the position vector of the ith particle.
    # return the distance between particle i and j according to the minimum
    # image convention.
	d = 0.0
    # <-- find d. -->
	dx = 0.0
	dy = 0.0
	dz = 0.0
	dx = (Ri[0] - Rj[0])%L
	dy = (Ri[1] - Rj[1])%L
	dz = (Ri[2] - Rj[2])%L
	dx = min(dx,L-dx)
	dy = min(dy,L-dy)
	dz = min(dz,L-dz)
	d = np.sqrt(dx*dx + dy*dy + dz*dz)
	return d

def Displacement(Ri, Rj):
    # Return the displacement of the ith particle relative to the jth 
    # particle according to the minimum image convention. Unlike the 'Distance'
    # function above, here you are returning a vector.
    dx = 0.0; dy = 0.0; dz = 0.0
    # <-- find dx, dy, and dz. -->
    dx = (Ri[0] - Rj[0])%L
    dy = (Ri[1] - Rj[1])%L
    dz = (Ri[2] - Rj[2])%L
    if( dx>(L-dx) ):
        dx = dx - L
    if( dy>(L-dy) ):
        dy = dy - L
    if( dz>(L-dz) ):
        dz = dz - L
    return [dx,dy,dz]
				
def AdatomSurfaceEnergy(Ri, Rs, E_as, r_as):
	'''
	Returns the potential energy of the 'adatom' due to the substrate.
	i is the index of the adatom being considered
	Rs is the list of substrate positions 
	E_as is the bonding energy or cohesive energy
		(epsilon in the Lennard-Jones equation)
	r_as is the nearest neighbor distance or equilibrium bond length 
		(r_m in the Lennard-Jones equation)
	'''
	Uappx = 0.0	
	for i in Rs:
		d = Distance(Ri, i)
		Uappx += 4*E_as*(r_as**12/d**6 - r_as**6/d**3)
	return Uappx


def AdatomAdatomEnergy(i, Ra, E_a, r_a):
	'''
	Returns the potential energy of the 'adatom' due to other 'adatoms'.
	i is the index of the adatom being considered
	Ra is the list of adatom positions 
	E_a is the bonding energy or cohesive energy
		(epsilon in the Lennard-Jones equation)
	r_a is the nearest neighbor distance or equilibrium bond length 
		(r_m in the Lennard-Jones equation)
	'''
	VLJ_a = 0.0	
	Ri = Ra[i]
	for j in range(len(Ra)):
		if(i!=j):
			Rj = Ra[j]
			r = Distance(Ri, Rj)
			VLJ_a += (4*E_a*((r_a/r)**12 - (r_a/r)**6))
	return VLJ_a
	
				
def VMDInit(N):
	with open('Trajectory.xyz', 'w') as outFile:
		outFile.write('%i\n'%N)

def VMDOut(R):
	with open('Trajectory.xyz', 'a') as outFile:
		for i in range(len(R)):
			ri = R[i]
			outFile.write('%i %.5f %.5f %.5f\n'%(i, ri[0], ri[1], ri[2]))

def AddAdatom(R, S):
	# R is the current adatoms, S is the substrate atoms
	adatom_bins = PutInBins(R)
	substrate_bins = PutInBins(substrate)
	max_y = D
	for x in adatom_bins:
		if len(x) > 0:
			for atom in x[-1]:
				if atom[1] > max_y:
					max_y = atom[1]
	new_x = random.random()*L
	Rn = [new_x, max_y]
	(nx, ny) = BinIndices(Rn)
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
	nearby_substrate = []
	nearby_adatoms = []
	for x in nearby_x:
		for y in nearby_y:
			if len(substrate_bins[x]) > y:
				for a in substrate_bins[x][y]:
					nearby_substrate.append(a)
			if len(adatom_bins) > x:
				if len(adatom_bins[x]) > y:
					for a in adatom_bins[x][y]:
						nearby_adatoms.append(a)
	print len(nearby_substrate), len(nearby_adatoms)
	

substrate = InitSubstrate()
adatoms = []

AddAdatom(adatoms, substrate)

for Ri in substrate:
	plt.scatter(Ri[0], Ri[1], color='blue')
# bins = PutInBins(R)
# for b in bins:
	# print [len(c) for c in b]
for x in range(nbins_x+1):
	plt.plot([x*bin_size, x*bin_size], [0, nbins_y*bin_size], color='red')
for y in range(nbins_y+1):
	plt.plot([0, nbins_x*bin_size], [y*bin_size, y*bin_size], color='red')
plt.show()