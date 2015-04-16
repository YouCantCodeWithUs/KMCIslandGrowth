#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import cm, axes

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
	v = np.matrix(v).transpose()
	rot_matrix = np.matrix([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
	v_rot = (rot_matrix*v).transpose()
	return np.array(v_rot)[0]

sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)
square_to_rhombus = np.matrix([[1, 0.5, 0], [0, sqrt3/2, 0], [0, 0, 1.0]])
rhombus_to_square = square_to_rhombus.getI()

def TransformSquareToRhombus(v):
	return np.array((square_to_rhombus*np.matrix(v).transpose()).transpose())[0]

def TransformRhombusToSquare(v):
	return np.array((rhombus_to_square*np.matrix(v).transpose()).transpose())[0]

def InitPositionsFCC(z_layers, xy_repeats, a):
	'''
	Initialize atoms in an FCC lattice with orientation (111) and inter-atom spacing `a`.
	Return the positions of these atoms.
	'''
	positions = []
	unit_cell = []
	column = []
	for z in range(3):
		atom = [a*z/3.0, a*z/3.0, -a*z*sqrt2/sqrt3]
		unit_cell.append(atom)
	unit_cell = np.array(unit_cell)
	for z in range(z_layers):
		column += list(unit_cell + np.array([0, 0, -a*z*sqrt2*sqrt3]))
	column = np.array(column)
	for x in range(xy_repeats):
		for y in range(xy_repeats):
			positions += list(column + np.array([a*x, a*y, 0.0]))
	positions = np.array(positions)
	for i in range(len(positions)):
		p = positions[i] - np.array([0.5, 0.5, 0])*xy_repeats*a
		positions[i] = TransformSquareToRhombus(p)
		# p = np.matrix(positions[i]).transpose()
		# positions[i] = np.array((square_to_rhombus*p).transpose())[0]
	return positions

def PutInBox(pos):
	'''
	Implement 2D rhombic periodic boundary conditions.
	Returns the 'correct' position.
	'''
	p = TransformRhombusToSquare(pos)
	L = r_s*substrate_repeats
	for n in range(2):
		# only x and y are periodic
		coord = p[n]%L
		if abs(coord) > L/2:
			coord += (-L if coord > 0 else L)
		if coord == L/2:
			coord = -L/2
		p[n] = coord
	return TransformSquareToRhombus(p)

def Displacement(Ri, Rj):
	'''
	Return the vector pointing from Ri to Rj according to the minimum image convention.
	'''
	dr = PutInBox(Rj-Ri)
	return dr

def Distance(Ri, Rj):
	'''
	Return the distance between particle i and particle j according to the minimum image convention
	'''	
	return np.sqrt(Distance2(Ri, Rj))
	
	'''
	MAKE SURE WE CONVERT d TO ANGSTROMS SOMEWHERE
	'''

def Distance2(Ri, Rj):
	d = Displacement(Ri, Rj)
	return np.dot(d, d)

def AdatomSurfaceEnergy(Ri, substrate, E_as, r_as):
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
	for atom in substrate:
		r2 = Distance2(Ri, atom)
		VLJ_as += 4*E_as*(r_as**12/r2**6 - r_as**6/r2**3)
	return VLJ_as

def AdatomSurfaceForce(Ri, substrate, E_as, r_as):
	FLJ_as = np.zeros(3)
	for a in substrate:
		r = Displacement(Ri, a)
		r2 = np.dot(r, r)
		r_mag = np.sqrt(r2)
		F = (4*E_as*(-12*r_as**12*r2**-6/r_mag + 6*r_as**6*r2**-3/r_mag))
		FLJ_as += r*F/r_mag
	return FLJ_as

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

def AdatomAdatomForce(i, Ra, E_a, r_a):
	FLJ_a = np.zeros(3)
	Ri = Ra[i]
	for j in range(len(Ra)):
		if i != j:
			Rj = Ra[j]
			r = Displacement(Ri, Rj)
			r2 = np.dot(r, r)
			r_mag = np.sqrt(r2)
			F = (4*E_a*(-12*r_a**12*r2**-6/r_mag + 6*r_a**6*r2**-3/r_mag))
			FLJ_a += r*F/r_mag
	return FLJ_a

def VerletNextR(r_t, v_t, a_t):
	return r_t + v_t*h + 0.5*a_t*h*h

def VerletNextV(v_t, a_t, a_t_plus_h):
	return v_t + 0.5*(a_t + a_t_plus_h)*h

def TimeStep(R, V):
	N = len(R)
	A  = np.zeros((N, 3))
	nR = np.zeros((N, 3))
	nV = np.zeros((N, 3))
	for i in range(len(R)):
		Fi = AdatomSurfaceForce(R[i], substrate, E_as, r_as)
		Fi += AdatomAdatomForce(i, R, E_a, r_a)
		A[i] = Fi/M
		nR[i] = VerletNextR(R[i], V[i], A[i])
		nR[i] = PutInBox(nR[i])
	for i in range(len(R)):
		nFi = AdatomSurfaceForce(nR[i], substrate, E_as, r_as)
		nFi += AdatomAdatomForce(i, nR, E_a, r_a)
		nAi = nFi/M
		nV[i] = VerletNextV(V[i], A[i], nAi)
	return (nR, nV)

def ComputeEnergy(R, V, substrate):
	U = sum([AdatomSurfaceEnergy(a, substrate, E_as, r_as) for a in R])
	KE = sum([0.5*M*sum([Vin*Vin for Vin in Vi]) for Vi in V])
	E = U+KE
	return E

def VMDInit(N):
	with open('Trajectory.xyz', 'w') as outFile:
		outFile.write('%i\n'%N)

def VMDOut(R):
	with open('Trajectory.xyz', 'a') as outFile:
		for i in range(len(R)):
			ri = R[i]
			outFile.write('%i %.5f %.5f %.5f\n'%(i, ri[0], ri[1], ri[2]))

substrate_thickness = 2
substrate_repeats = 5
substrate = InitPositionsFCC(substrate_thickness, substrate_repeats, r_s)
top_layer = [s for s in substrate if s[2] == 0]
# there is an atom at 0,0,0 when substrate_repeats is even
# seems like we can get away with just 2 repeats, but require thickness of 2 (x3)

# VMD breaks at larger than (3, 3, a)
# Vesta handles bigger systems
# VMDInit(substrate)
# VMDOut(substrate)
# VMDOut(substrate)

M = 48.0
h = 0.1

R = []
V = []
for i in range(5):
	for j in range(2):
		a = np.array([i*r_s, j*r_s, r_as])
		a = TransformSquareToRhombus(a)
		a = PutInBox(a)
		R.append(a)
		v = np.array([random.random()-0.5, random.random()-0.5, random.random()-0.5])
		v *= 0.1
		V.append(v)
R = np.array(R)
V = np.array(V)

VMDInit(len(top_layer) + len(R))
VMDOut(top_layer + list(R))
# Es = []
for t in range(1000):
	if t%100 == 0:
		E = ComputeEnergy(R, V, substrate)
		print t, E
	(nR, nV) = TimeStep(R, V)
	R = nR.copy()
	V = nV.copy()
	# Es.append(E)
	VMDOut(top_layer + list(R))

# plt.plot(Es)
# plt.show()

# ts = np.linspace(0, 2*np.pi, 60)
# rs = np.linspace(0, r_s, 40)
# for z in range(1, 10):
# 	z *= r_as
# 	Es = []
# 	for r in rs:
# 		E_row = []
# 		for t in ts:
# 			adatom = np.array([r*np.cos(t), r*np.sin(t), z])
# 			E = AdatomSurfaceEnergy(adatom, substrate, E_as, r_as)
# 			E_row.append(E)
# 		Es.append(E_row)
# 	E_min = min([min(e) for e in Es])
# 	E_max = max([max(e) for e in Es])
# 	ax = plt.subplot(111, polar=True)
# 	CS = plt.contour(ts, rs, Es, levels=np.linspace(E_min, E_max, 21), cmap=cm.jet)
# 	CB = plt.colorbar(CS, extend='both')
# 	plt.savefig('polar-%.1f.png'%(z), dpi=150)
# 	plt.clf()

