#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
from multiprocessing import Pool
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

def SurfaceAtoms(adatoms, substrate_bins):
	'''
	Identifies surface adatoms. 
	Surface atoms are defined as adatoms coordinated by < 5 other atoms.
	adatoms: np.array(np.array[x, y]) - position of adatoms
	
	returns: List[np.array[x, y]] - list of positions of surface adatoms.
	'''
	surfaceAtoms = []
	(nearest_adatoms, nearest_substrate) = Bins.NearestNeighbors(adatoms, substrate_bins)
	for i in range(len(adatoms)):
		if len(nearest_adatoms[i]) + len(nearest_substrate[i]) < 5:
			surfaceAtoms.append(adatoms[i])
	return surfaceAtoms

def DepositAdatom(adatoms, substrate_bins):
	'''
	adatoms: np.array(np.array[x, y]) - position of adatoms
	
	returns: np.array(np.array[x, y]) - position of adatoms with additional adatom
	'''
	min_y = 0
	if len(adatoms) > 0:
		min_y = min([p[1] for p in adatoms])
	Ri = Periodic.PutInBox(np.array([random.random()*gv.L, 2*gv.r_as + min_y]))
	adatom_bins = Bins.PutInBins(adatoms)
	closeby = False
	while not closeby:
		nearby_adatoms = Bins.NearbyAtoms(Ri, adatom_bins)
		for Rj in nearby_adatoms:
			d = Periodic.Distance(Ri, Rj)
			if d < 2*gv.r_a:
				closeby = True
		nearby_substrate = Bins.NearbyAtoms(Ri, substrate_bins)
		for Rj in nearby_substrate:
			d = Periodic.Distance(Ri, Rj)
			if d < 2*gv.r_as:
				closeby = True
		if not closeby:
			Ri -= np.array([0, gv.r_a/2])
	i = len(adatoms)
	adatoms.append(Ri)
	adatoms = RelaxAdatoms(adatoms, substrate_bins, around=i)
	return adatoms

def RelaxAdatoms(adatoms, substrate_bins, around=None):
	backup = adatoms[:]
	N = len(adatoms)
	nearby_indices = []
	relaxing_adatoms = []
	if around != None:
		adatom_bins = Bins.PutInBins(adatoms)
		nearby_adatoms = Bins.NearbyAtoms(adatoms[around], adatom_bins)
		truth = zip(*[iter(np.in1d(adatoms, nearby_adatoms))]*2)
		truth = [True if t == (True, True) else False for t in truth]
		nearby_indices = []
		for i in range(N):
			if truth[i]:
				nearby_indices.append(i)
		nearby_indices = sorted(nearby_indices, reverse=True)
		for i in nearby_indices:
			relaxing_adatoms.append(adatoms.pop(i))
	else:
		relaxing_adatoms = adatoms[:]
	p = Pool(4)
	xn = relaxing_adatoms[:]
	N = len(xn)
	xnp = []
	dx0 = [Energy.AdatomAdatomForce(i, xn) + Energy.AdatomSurfaceForce(xn[i], substrate_bins) for i in range(N)]
	# pos = [xn[N-1]]
	# Conjugate Gradient
	for i in range(N):
		other_adatoms = xn[:]
		other_adatoms.pop(i)
		xs = [(xn[i] + a*dx0[i]/10, other_adatoms, substrate_bins) for a in range(10)]
		ys = p.map(Energy.RelaxEnergy, xs)
		# ys = [Energy.ArbitraryAdatomEnergy(x, other_adatoms, E_a, r_a) + Energy.AdatomSurfaceEnergy(x, substrate_bins, E_as, r_as) for x in xs]
		(xmin, a, a) = xs[ys.index(min(ys))]
		xs = [(xmin - dx0[i]/15 + a*dx0[i]/50, other_adatoms, substrate_bins) for a in range(10)]
		ys = p.map(Energy.RelaxEnergy, xs)
		# ys = [Energy.ArbitraryAdatomEnergy(x, other_adatoms, E_a, r_a) + Energy.AdatomSurfaceEnergy(x, substrate_bins, E_as, r_as) for x in xs]
		(xmin, a, a) = xs[ys.index(min(ys))]
		xnp.append(xmin)
	sn = dx0[:]
	snp = dx0[:]
	dxnp = [Energy.AdatomAdatomForce(i, xnp) + Energy.AdatomSurfaceForce(xnp[i], substrate_bins) for i in range(N)]
	maxF = max([np.dot(dxi, dxi) for dxi in dxnp])
	numRelaxations = 1
	while maxF > 1e-4:
		lastMaxF = maxF
		for i in range(N):
			top = np.dot(xnp[i], (xnp[i] - xn[i]))
			bottom = np.dot(xn[i], xn[i])
			beta = top/bottom
			snp[i] = dxnp[i] + beta*sn[i]
			
			x = xn[N-i-1]
			if x[1] > gv.Dmax or x[1] < 0:
				print 'REVERTING TO BACKUP A'
				if around != None:
					backup.pop(around)
				return RelaxAdatoms(backup, substrate_bins, len(backup)-1)
		xn = xnp[:]
		sn = snp[:]
		for i in range(N):
			other_adatoms = xn[:]
			other_adatoms.pop(i)
			xs = [(xn[i] + a*sn[i]/10, other_adatoms, substrate_bins) for a in range(10)]
			ys = p.map(Energy.RelaxEnergy, xs)
			# ys = [Energy.ArbitraryAdatomEnergy(x, other_adatoms, E_a, r_a) + Energy.AdatomSurfaceEnergy(x, substrate_bins, E_as, r_as) for x in xs]
			(xmin, a, a) = xs[ys.index(min(ys))]
			xs = [(xmin - sn[i]/15 + a*sn[i]/50, other_adatoms, substrate_bins) for a in range(10)]
			ys = p.map(Energy.RelaxEnergy, xs)
			# ys = [Energy.ArbitraryAdatomEnergy(x, other_adatoms, E_a, r_a) + Energy.AdatomSurfaceEnergy(x, substrate_bins, E_as, r_as) for x in xs]
			(xmin, a, a) = xs[ys.index(min(ys))]
			xmin = Periodic.PutInBox(xmin)
			xnp[i] = xmin
		xn = xnp[:]
		# pos.append(xn[N-1])
		dxnp = [Energy.AdatomAdatomForce(i, xn) + Energy.AdatomSurfaceForce(xn[i], substrate_bins) for i in range(N)]
		maxF = max([np.dot(dxi, dxi) for dxi in dxnp])
		numRelaxations += 1
		print maxF
		if numRelaxations > 400:
			print 'REVERTING TO BACKUP B'
			if around != None:
				backup.pop(around)
			return RelaxAdatoms(backup, substrate_bins, len(backup)-1)
	
	willRevert = False
	for i in range(N):
		x = xn[N-i-1]
		if x[1] > gv.Dmax or x[1] < 0:
			print 'REVERTING TO BACKUP C'
			if around != None:
				backup.pop(around)
			return RelaxAdatoms(backup, substrate_bins, len(backup)-1)
	# xs = [p[0] for p in pos]
	# ys = [p[1] for p in pos]
	# plt.plot(xs, ys)
	# plt.show()
	adatoms += xn
	return adatoms

def HoppingRates(adatoms, substrate_bins):
	omega = 1.0#e12
	surf = SurfaceAtoms(adatoms, substrate_bins, gv.r_as, gv.r_a)
	indexes = [adatoms.index(s) for s in surf]
	return [omega*np.exp(Energy.DeltaU(i, adatoms, substrate_bins, E_as, r_as, E_a, r_a)*betaK) for i in indexes]

def HoppingPartialSums(Rk):
	return [sum(Rk[:i+1]) for i in range(len(Rk))]

def TotalRate(Rd, Rk):
	return Rd + sum(Rk)

def PlotSubstrate(substrate, color='blue'):
	PlotAtoms(substrate, color)
	for x in range(gv.nbins_x + 2):
		plt.plot([x*gv.bin_size - gv.L/2, x*gv.bin_size - gv.L/2], [gv.Dmin, gv.nbins_y*gv.bin_size + gv.Dmin], color='red')
	for y in range(gv.nbins_y + 1):
		plt.plot([-gv.L/2, (gv.nbins_x + 1)*gv.bin_size - gv.L/2], [y*gv.bin_size + gv.Dmin, y*gv.bin_size + gv.Dmin], color='red')
	# plt.show()

def PlotAtoms(atoms, color='blue'):
	for a in atoms:
		plt.scatter(a[0], a[1], color=color)

substrate = InitSubstrate(gv.W, gv.Hs, gv.r_s)
substrate_bins = Bins.PutInBins(substrate)

adatoms = []
for i in range(50):
	print i
	adatoms = DepositAdatom(adatoms, substrate_bins)
	# if i % 5 == 0:
	PlotSubstrate(substrate, 'blue')
	PlotAtoms(adatoms, 'green')
	surf = SurfaceAtoms(adatoms, substrate_bins)
	PlotAtoms(surf, 'red')
	plt.savefig('_Images/%2i.png'%i)
	# plt.show()
	plt.clf()

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

# PlotSubstrate(substrate)
# surf = SurfaceAtoms(adatoms, substrate_bins, gv.r_as, gv.r_a)
# PlotSubstrate(surf, 'red')
# plt.show()
# plt.clf()

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