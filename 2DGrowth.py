#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
from multiprocessing import Pool
from matplotlib import pyplot as plt
import pprint

import Constants as gv
import Periodic
import Bins
import Energy

pool = Pool(1)

def InitSubstrate():
	'''
	Initialize substrate atom positions in a hexagonal grid below y = 0
	
	w:   Int   - number of atoms in a horizontal row
	h:   Int   - number of rows
	r_s: Float - spacing between atoms
	
	returns: np.array(np.array[x, y]) - list of substrate atom positions
	'''
	R = []
	roffset = gv.r_s/2
	for i in range(gv.Hs):
		y = -i*gv.r_s*gv.sqrt3/2
		for x in range(gv.W):
			Ri = x*gv.r_s
			if i%2 == 1:
				Ri += roffset
			R.append(Ri); R.append(y)
	return Periodic.PutAllInBox(np.array(R))

def DepositionRate():
	'''
	Returns a calculated value based on estimated wafer size and flux rate.
	'''
	return 6.0*18/gv.W*gv.deposition_rate
	return float(gv.W)*MLps

def SurfaceAtoms(adatoms, substrate_bins):
	'''
	Identifies surface adatoms. 
	Surface atoms are defined as adatoms coordinated by < 5 other atoms.
	
	adatoms: np.array(np.array[x, y]) - position of adatoms
	substrate_bins: list(list(np.array[x, y])) - bin'd position of substrate atoms
	
	returns: List[np.array[x, y]] - list of positions of surface adatoms.
	'''
	surfaceAtoms = []
	(nearest_adatoms, nearest_substrate) = Bins.NearestNeighbors(adatoms, substrate_bins)
	for i in range(len(adatoms)/2):
		if len(nearest_adatoms[i])/2 + len(nearest_substrate[i])/2 < 5:
			surfaceAtoms.append(adatoms[2*i])
			surfaceAtoms.append(adatoms[2*i+1])
	return surfaceAtoms

def DepositAdatom(adatoms, substrate_bins):
	'''
	Deposits new adatom onto the surface, then performs a relaxation.
	
	adatoms: np.array(np.array[x, y]) - position of adatoms
	substrate_bins: list(list(np.array[x, y])) - bin'd position of substrate atoms
	
	returns: np.array(np.array[x, y]) - position of adatoms with additional adatom
	'''
	min_y = 0
	if len(adatoms) > 0:
		min_y = min([adatoms[2*i+1] for i in range(len(adatoms)/2)])
	new_x = Periodic.PutInBox(random.random()*gv.L)
	new_y = 2*gv.r_as + min_y
	adatom_bins = Bins.PutInBins(adatoms)
	closeby = False
	while not closeby:
		Ri = np.array([new_x, new_y])
		nearby_adatoms = Bins.NearbyAtoms(new_x, new_y, adatom_bins)
		d = Periodic.Distances(Ri, nearby_adatoms)[0]
		if len(d) > 0 and min(d) < 2*gv.r_a:
			closeby = True
		
		nearby_substrate = Bins.NearbyAtoms(new_x, new_y, substrate_bins)
		d = Periodic.Distances(Ri, nearby_substrate)[0]
		if len(d) > 0 and min(d) < 2*gv.r_as:
			closeby = True
		if not closeby:
			new_y -= gv.r_a/2
	adatoms = np.append(adatoms, Ri)
	adatoms = LocalRelaxation(adatoms, substrate_bins, Ri)
	return adatoms

def UnzipPositions(l):
	L = []
	for i in l:
		L.append(i[0]); L.append(i[1])
	return np.array(L)

def ZipPositions(l):
	return [np.array([l[2*i], l[2*i+1]]) for i in range(len(l)/2)]

def Relaxation(adatoms, substrate_bins, scale=1, threshold=1e-4):
	'''
	Relaxes the deposited adatoms into the lowest energy position using a conjugate gradient algorithm.
	Runs recursively if the step size is too small.
	Uses a multiprocessing pool for extra speed.
	
	adatoms: np.array(np.array[x, y]) - position of adatoms
	substrate_bins: list(list(np.array[x, y])) - bin'd position of substrate atoms
	scale: int - How much to scale the step size down by
	
	returns: np.array(np.array[x, y]) - position of adatoms with additional adatom
	'''
	global pool
	# If the energy of a proposed relaxed arrangement exceeds this Ui, then halve the stepsize and start over.
	Ui = Energy.TotalEnergy(adatoms, substrate_bins)
	N = len(adatoms)/2
	xn = adatoms.copy()
	# Initial forces on each atom
	dx0 = Energy.AdatomAdatomForces(xn) + Energy.AdatomSubstrateForces(xn, substrate_bins)
	xs = [(adatoms + a*dx0/10/scale, substrate_bins) for a in range(20)]
	ys = pool.map(Energy.RelaxEnergy, xs)
	(xnp, a) = xs[ys.index(min(ys))]
	sn = dx0[:]
	snp = dx0[:]
	# Force on each atom in the lowest energy position
	dxnp = Energy.AdatomAdatomForces(xnp) + Energy.AdatomSubstrateForces(xnp, substrate_bins)
	maxF = max([np.dot(dxnp[2*i:2*i+2], dxnp[2*i:2*i+2]) for i in range(len(dxnp))])
	lastmaxF = maxF
	while maxF > threshold:
		# Calculate the conjugate direction step size, beta
		top = np.dot(xnp, (xnp - xn))
		bot = np.dot(xn, xn)
		beta = top/bot
		# Update conjugate direction
		snp = dxnp + beta*sn
		
		xn = xnp.copy()
		sn = snp.copy()
		
		xs = [(xn + a*sn/10/scale, substrate_bins) for a in range(20)]
		ys = pool.map(Energy.RelaxEnergy, xs)
		plt.plot(ys); plt.show()
		(xmin, a) = xs[ys.index(min(ys))]
		
		# Calculate forces at new lowest energy position
		dxnp = Energy.AdatomAdatomForces(xmin) + Energy.AdatomSubstrateForces(xmin, substrate_bins)
		maxF = max([np.dot(dxnp[2*i:2*i+2], dxnp[2*i:2*i+2]) for i in range(len(dxnp))])
		print maxF
		if abs(lastmaxF - maxF) < 1e-6:
			print 'changing scale', scale*2, maxF
			return Relaxation(adatoms, substrate_bins, scale=scale*2, threshold=threshold)
		lastmaxF = maxF
	return xnp

def LocalRelaxation(adatoms, substrate_bins, around):
	'''
	Relaxes the deposited adatoms into the lowest energy position using a conjugate gradient algorithm.
	Performs an additional global relaxation is global forces are too large.
	
	adatoms: np.array(np.array[x, y]) - position of adatoms
	substrate_bins: list(list(np.array[x, y])) - bin'd position of substrate atoms
	around: np.array[x, y] - position around which to perform the relaxation
	
	returns: np.array(np.array[x, y]) - positions of relaxed adatoms
	'''
	
	nearby_indices = []
	relaxing_adatoms = []
	adatom_bins = Bins.PutInBins(adatoms)
	nearby_adatoms = Bins.NearbyAtoms(around[0], around[1], adatom_bins)
	nearby_indices = [list(Ds).index(0) for Ds in Periodic.Distances(nearby_adatoms, adatoms)]
	nearby_indices = sorted(nearby_indices, reverse=True)
	for i in nearby_indices:
		relaxing_adatoms.append(adatoms[2*i])
		relaxing_adatoms.append(adatoms[2*i+1])
		adatoms = np.delete(adatoms, 2*i)
		adatoms = np.delete(adatoms, 2*i)
	relaxing_adatoms = np.array(relaxing_adatoms)
	relaxing_adatoms = Relaxation(relaxing_adatoms, substrate_bins, scale=4)
	adatoms = np.append(adatoms, relaxing_adatoms)
	sys.exit()
	forces = [Energy.AdatomAdatomForce(i, adatoms) + Energy.AdatomSurfaceForce(adatoms[i], substrate_bins) for i in range(len(adatoms))]
	maxF = max([np.dot(f, f) for f in forces])
	if maxF > 1e-3:
		print 'global required'
		adatoms = Relaxation(adatoms, substrate_bins, scale=4, threshold=1e-4)
	return adatoms

def HoppingRates(adatoms, substrate_bins):
	'''
	Calculates the hopping rate of each surface adatom.
	
	adatoms: np.array(np.array[x, y]) - position of adatoms
	substrate_bins: list(list(np.array[x, y])) - bin'd position of substrate atoms
	
	returns: list(float) - hopping rate of each surface atom
	'''
	omega = 1.0e12
	surf = SurfaceAtoms(adatoms, substrate_bins)
	truth = zip(*[iter(np.in1d(adatoms, surf))]*2)
	truth = [True if t == (True, True) else False for t in truth]
	surf_indices = []
	for i in range(len(surf)):
		if truth[i]:
			surf_indices.append(i)
	return [omega*np.exp(Energy.DeltaU(i, adatoms, substrate_bins)*gv.beta) for i in surf_indices]

def HoppingPartialSums(Rk):
	'''
	Cumulative sum hopping rates. Used to determine whether to hop or deposit.
	'''
	return [0] + [sum(Rk[:i+1]) for i in range(len(Rk))]

def TotalRate(Rd, Rk):
	'''
	Addition!
	'''
	return Rd + sum(Rk)

def HopAtom(i, adatoms, substrate_bins):
	'''
	Moves a surface adatom and to a nearby spot. Performs a local relaxation around the new position.
	
	i: int - index of adatom to hop
	adatoms: np.array(np.array[x, y]) - position of adatoms
	substrate_bins: list(list(np.array[x, y])) - bin'd position of substrate atoms
	
	returns: np.array(np.array[x, y]) - positions of relaxed adatoms after hop
	'''
	surf = SurfaceAtoms(adatoms, substrate_bins)
	jumper = adatoms.pop(i)
	surf = [s for s in surf if Periodic.Distance(jumper, s) < 3*gv.r_a]
	jump_to_surf = surf[random.randint(0, len(surf)-1)]
	jump_directions = [i*np.pi/3 for i in range(6)]
	jump_vectors = [np.array([np.cos(t), np.sin(t)])*gv.r_a for t in jump_directions]
	jump_positions = [jump_to_surf + v for v in jump_vectors]
	jump_energies = [Energy.ArbitraryAdatomEnergy(p, adatoms) + Energy.AdatomSurfaceEnergy(p, substrate_bins) for p in jump_positions]
	jump_to_pos = jump_positions[jump_energies.index(min(jump_energies))]
	jump_to_pos = Periodic.PutInBox(jump_to_pos)
	adatoms.append(jump_to_pos)
	adatoms = LocalRelaxation(adatoms, substrate_bins, jump_to_pos)
	return adatoms

def PlotSubstrate(substrate, color='blue'):
	'''
	Pretty plots.
	'''
	PlotAtoms(substrate, color)
	for x in range(gv.nbins_x + 2):
		plt.plot([x*gv.bin_size - gv.L/2, x*gv.bin_size - gv.L/2], [gv.Dmin, gv.nbins_y*gv.bin_size + gv.Dmin], color='red')
	for y in range(gv.nbins_y + 1):
		plt.plot([-gv.L/2, (gv.nbins_x + 1)*gv.bin_size - gv.L/2], [y*gv.bin_size + gv.Dmin, y*gv.bin_size + gv.Dmin], color='red')
	# plt.show()

def PlotAtoms(atoms, color='blue'):
	'''
	Pretty plots.
	'''
	for i in range(len(atoms)/2):
		plt.scatter(atoms[2*i], atoms[2*i+1], color=color)

def PlotEnergy(adatoms, substrate_bins):
	'''
	Pretty plots.
	'''
	ymin = min(a[1] for a in adatoms) + gv.r_a
	xs = np.linspace(-gv.L/2, gv.L/2, 500)
	pos = [(np.array([x, ymin]), adatoms, substrate_bins) for x in xs]
	Es = pool.map(Energy.RelaxEnergy, pos)
	plt.plot(xs, Es)
	plt.axis([min(xs), max(xs), min(Es)-0.1, max(Es)+0.1])
	# plt.show()
	plt.savefig(gv.folder + 'E%.2i.png'%t)
	plt.clf()

substrate = InitSubstrate()
substrate_bins = Bins.PutInBins(substrate)

surf_substrate = SurfaceAtoms(substrate, substrate_bins)

adatoms = np.array([])
adatoms = InitSubstrate()[:gv.W*2]
for i in range(len(adatoms)/2):
	adatoms[2*i] -= gv.r_a/2
	adatoms[2*i+1] += gv.r_a*gv.sqrt3/2
adatoms = DepositAdatom(adatoms, substrate_bins)
# adatoms = Relaxation(adatoms, substrate_bins, scale=2)
# t = 0
# while True:
# 	Rk = HoppingRates(adatoms, substrate_bins)
# 	pj = HoppingPartialSums(Rk)
# 	Rd = DepositionRate()
# 	Rtot = TotalRate(Rd, Rk)
# 	r = random.random()*Rtot
# 	print t, pj[-1], Rd, r
# 	HoppingAtom = [p for p in pj if p > r]
# 	if len(HoppingAtom) > 0:
# 		print 'HOP'
# 		adatoms = HopAtom(pj.index(HoppingAtom[0])-1, adatoms, substrate_bins)
# 	else:
# 		adatoms = DepositAdatom(adatoms, substrate_bins)
#
# 	# if i % 5 == 0:
# 	PlotSubstrate(substrate, 'blue')
# 	PlotAtoms(adatoms, 'green')
# 	surf = SurfaceAtoms(adatoms, substrate_bins)
# 	PlotAtoms(surf, 'red')
# 	plt.axis([-gv.L/2-0.1, gv.L/2+0.1, gv.Dmin-0.1, gv.Dmax+3*gv.r_as])
# 	plt.savefig(gv.folder + '%.2i.png'%t)
# 	# plt.show()
# 	plt.clf()
# 	# PlotEnergy(adatoms, substrate_bins)
# 	t += 1

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