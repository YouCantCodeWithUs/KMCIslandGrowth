#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
import numpy as np

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

def AdatomSubstrateEnergy(adatom, substrate):
	'''
	Returns the potential energy of the `adatom` due to the `substrate` atoms
	'''

def AdatomAdatomEnergy(adatom,i):
	'''
	Returns the potential energy of the 'adatom' due to other 'adatoms'.
	'''

def VMDInit(N):
	with open('Trajectory.xyz', 'w') as outFile:
		outFile.write('%i\n'%N)

def VMDOut(R):
	with open('Trajectory.xyz', 'a') as outFile:
		for i in range(len(R)):
			ri = R[i]
			outFile.write('%i %.5f %.5f %.5f\n'%(i, ri[0], ri[1], ri[2]))

a = 1.0
substrate = InitPositionsFCC(3, 1, a)
# VMD breaks at larger than (3, 3, a)
VMDInit(len(substrate))
VMDOut(substrate)
VMDOut(substrate)
