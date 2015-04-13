#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
import numpy as np

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

def rotate(v, t):
	'''
	Rotate the vector `v` by `t` radians and return the result.
	'''
	rot_matrix = [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
	return [r[0] for r in rot_matrix*v]

def InitPositionsFCC(layers, radius, a):
	'''
	Initialize atoms in an FCC lattice with orientation (111) and spacing `a`.
	Return the positions of these atoms.
	'''
	# 0,0,0 is the central atom (hexagonal boundary conditions)
	positions = []
	for l in range(layers):
		layer = []
		z = -a*np.sqrt(11.0/12)*l
		layer.append([-a/3.0,0.0,z])
		for t in range(6):
			theta = t*np.pi/3
			for r1 in range(1, radius+1):
				r = r1 * a
				x = r*np.cos(theta) - a/3.0; y = r*np.sin(theta)
				new = [x, y, z]
				for r2 in range(r1):
					unit = np.array([1.0,0.0,0.0])
					offset = rotate(unit, (t+2)*np.pi/3)
					offset = np.dot(r2, offset)
					layer.append(list(new + offset))
		if l%3 == 1:
			unit = np.array([a/np.sqrt(3), 0, 0])
			offset = rotate(unit, np.pi/6)
			for i in range(len(layer)):
				layer[i] = list(np.array(layer[i]) + offset)
		elif l%3 == 2:
			unit = np.array([a/np.sqrt(3), 0, 0])
			offset = rotate(unit, -np.pi/6)
			for i in range(len(layer)):
				layer[i] = list(np.array(layer[i]) + offset)
		positions += layer
	return np.array(positions)

'''
Hexagon border:
a = 'radius'
t = theta mod pi/3
r(theta) = 2 a / (sin t + sqrt(3) cos t)
'''

def Distance(Ri, Rj):
	'''
	Return the distance between particle i and particle j according to the
	minimum image convention
	'''	
	return d
	
	'''
	MAKE SURE WE CONVERT d TO ANGSTROMS SOMEWHERE
	'''
	
def AdatomSurfaceEnergy(i, Rs, E_as, r_as):
	'''
	Returns the potential energy of the 'adatom' due to the substrate.
	i is the index of the adatom being considered
	Rs is the list of substrate positions 
	E_as is the bonding energy or cohesive energy
		(epsilon in the Lennard-Jones equation)
	r_as is the nearest neighbor distance or equilibrium bond length 
		(r_m in the Lennard-Jones equation)
	'''
	VLJ_as = 0.0	
	for j in Rs:
		r = Distance(i,j)
		VLJ_as += (4*E_as*((r_as/r)**12 - 2*(r_as/r)**6))
	return VLJ_as


def AdatomEnergy(i, Ra, E_a, r_a):
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
	for j in Ra:
		if(i!=j):
			r = Distance(i,j)
			VLJ_a += (4*E_a*((r_a/r)**12 - 2*(r_a/r)**6))
	return VLJ_a

def VMDInit(N):
	with open('Trajectory.xyz', 'w') as outFile:
		outFile.write('%i\n'%N)

def VMDOut(R):
	with open('Trajectory.xyz', 'a') as outFile:
		for i in range(len(R)):
			ri = R[i]
			outFile.write('%i %.5f %.5f %.5f\n'%(i, ri[0], ri[1], ri[2]))

a = 1.0
R = InitPositionsFCC(3, 1, a)
# VMD breaks at larger than (3, 3, a)
VMDInit(len(R))
VMDOut(R)
VMDOut(R)
# print R