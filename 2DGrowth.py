#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
from matplotlib import pyplot as plt

import Constants as gv
import Periodic
import Bins
import Energy

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
		y = -i*r_s*gv.sqrt3/2
		for x in range(w):
			Ri = np.array([x*r_s, y])
			if i%2 == 1:
				Ri += roffset
			R.append(Periodic.PutInBox(Ri))
	return np.array(R)

def DepositionRate(MLps):
	return float(gv.W)*MLps

def SurfaceAtoms(adatoms, substrate_bins, r_as, r_a):
	'''
	Identifies surface adatoms. 
	Surface atoms are defined as adatoms coordinated by < 5 other atoms.
	adatoms: np.array(np.array[x, y]) - position of adatoms
	
	returns: List[np.array[x, y]] - list of positions of surface adatoms.
	'''
	surfaceAtoms = []
	(nearest_adatoms, nearest_substrate) = Bins.NearestNeighbors(adatoms, substrate_bins, r_as, r_a)
	for i in range(len(adatoms)):
		if len(nearest_adatoms[i]) + len(nearest_substrate[i]) < 5:
			surfaceAtoms.append(adatoms[i])
	# Filter out adatom-substrate interface
	return surfaceAtoms

def DepositAdatom(adatoms, substrate_bins, E_as, r_as, E_a, r_a):
	'''
	adatoms: np.array(np.array[x, y]) - position of adatoms
	
	returns: np.array(np.array[x, y]) - position of adatoms with additional adatom
	'''
	min_height = 0
	if len(adatoms) > 0:
		min_height = min([a[1] for a in adatoms])
	new_adatom = np.array(Periodic.PutInBox(np.array([random.random()*gv.L, 1.5*r_as + min_height])))
	i = len(adatoms)
	adatoms.append(new_adatom)
	adatoms = RelaxAdatom(i, adatoms, substrate_bins, E_as, r_as, E_a, r_a)
	return adatoms

def RelaxAdatom(i, adatoms, substrate_bins, E_as, r_as, E_a, r_a):
	F = np.array([1, 1])
	e = 0
	# ri = adatoms[i]
	# positions = []
	while np.dot(F, F) > 1e-4:
		# positions.append(adatoms[i])
		F = Energy.AdatomAdatomForce(i, adatoms, E_a, r_a) + Energy.AdatomSurfaceForce(adatoms[i], substrate_bins, E_as, r_as)
		xs = [adatoms[i] + F*j/10/1.1**e for j in range(10)]
		ys = [Energy.ArbitraryAdatomEnergy(x, adatoms[:i], E_a, r_a) + Energy.AdatomSurfaceEnergy(x, substrate_bins, E_as, r_as) for x in xs]
		relaxed_pos = xs[ys.index(min(ys))]
		adatoms[i] = relaxed_pos
		e += 1
		if e > 30:
			# print e
			e = 5
	# print e
	# plt.scatter([p[0] for p in positions], [p[1] for p in positions])
	# plt.plot([p[0] for p in positions], [p[1] for p in positions])
	# plt.savefig('falling.png')
	return adatoms

def HoppingRates(adatoms, substrate_bins, E_as, r_as, E_a, r_a, betaK):
	omega = 1.0#e12
	return [omega*np.exp(Energy.DeltaU(i, adatoms, substrate_bins, E_as, r_as, E_a, r_a)*betaK) for i in range(len(adatoms))]

def HoppingPartialSums(Rk):
	return [sum(Rk[:i+1]) for i in range(len(Rk))]

def TotalRate(Rd, Rk):
	return Rd + sum(Rk)

def PlotSubstrate(substrate, color='blue'):
	for a in substrate:
		plt.scatter(a[0], a[1], color=color)
	for x in range(gv.nbins_x + 2):
		plt.plot([x*gv.bin_size - gv.L/2, x*gv.bin_size - gv.L/2], [gv.Dmin, gv.nbins_y*gv.bin_size + gv.Dmin], color='red')
	for y in range(gv.nbins_y + 1):
		plt.plot([-gv.L/2, (gv.nbins_x + 1)*gv.bin_size - gv.L/2], [y*gv.bin_size + gv.Dmin, y*gv.bin_size + gv.Dmin], color='red')
	# plt.show()

substrate = InitSubstrate(gv.W, gv.Hs, gv.r_s)
substrate_bins = Bins.PutInBins(substrate)

adatoms = []
for i in range(1):
	adatoms = DepositAdatom(adatoms, substrate_bins, gv.E_as, gv.r_as, gv.E_a, gv.r_a)

# Rk = HoppingRates(adatoms, substrate_bins, gv.E_as, gv.r_as, gv.E_a, gv.r_a, gv.beta/gv.boltzmann)
# pj = HoppingPartialSums(Rk)
# Rd = DepositionRate(gv.deposition_rate)
# Rtot = TotalRate(Rd, Rk)
# r = random.random()*Rtot
# HoppingAtom = [p for p in pj if p > r]
# if len(HoppingAtom) > 0:
# 	# hop atom
# 	print 'HOP'
# else:
# 	adatoms = DepositAdatom(adatoms, substrate_bins, gv.r_as)
# print adatoms

PlotSubstrate(substrate)
surf = SurfaceAtoms(adatoms, substrate_bins, gv.r_as, gv.r_a)
PlotSubstrate(surf, 'red')
plt.show()
plt.clf()

# minimum energy position
# xs = np.linspace(0, gv.L, 500)
# Es = [Energy.AdatomSurfaceEnergy(np.array([x, gv.r_as]), substrate_bins, gv.E_as, gv.r_as) for x in xs]
# xmin = xs[Es.index(min(Es))]
# xs = np.linspace(xmin-0.1, xmin+0.1, 500)
# Es = [Energy.AdatomSurfaceEnergy(np.array([x, gv.r_as]), substrate_bins, gv.E_as, gv.r_as) for x in xs]
# xmin = xs[Es.index(min(Es))]
# ys = np.linspace(gv.r_as/2, gv.r_as*2, 500)
# Es = [Energy.AdatomSurfaceEnergy(np.array([xmin, y]), substrate_bins, gv.E_as, gv.r_as) for y in ys]
# ymin = ys[Es.index(min(Es))]
# ys = np.linspace(ymin-0.1, ymin+0.1, 500)
# Es = [Energy.AdatomSurfaceEnergy(np.array([xmin, y]), substrate_bins, gv.E_as, gv.r_as) for y in ys]
# ymin = ys[Es.index(min(Es))]
# print xmin/gv.r_as, ymin/gv.r_as
# print Energy.AdatomSurfaceEnergy(np.array([xmin, ymin]), substrate_bins, gv.E_as, gv.r_as)
# print Energy.AdatomSurfaceForce(np.array([xmin, ymin]), substrate_bins, gv.E_as, gv.r_as)
# plt.plot(ys, Es)
# plt.show()