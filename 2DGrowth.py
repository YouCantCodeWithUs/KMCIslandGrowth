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
	roffset = np.array([gv.r_s/2, 0])
	for i in range(gv.Hs):
		y = -i*gv.r_s*gv.sqrt3/2
		for x in range(gv.W):
			Ri = np.array([x*gv.r_s, y])
			if i%2 == 1:
				Ri += roffset
			R.append(Periodic.PutInBox(Ri))
	return np.array(R)

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
	for i in range(len(adatoms)):
		if len(nearest_adatoms[i]) + len(nearest_substrate[i]) < 5:
			surfaceAtoms.append(adatoms[i])
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
	adatoms = LocalRelaxation(adatoms, substrate_bins, Ri)
	return adatoms

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
	backup = adatoms[:]
	if scale > 1024:
		if threshold > 1e-1:
			print 'I GIVE UP'
			sys.exit()
		print 'GAH!', threshold*5
		return Relaxation(backup, substrate_bins, scale=16, threshold=threshold*5)
	# If the energy of a proposed relaxed arrangement exceeds this Ui, then halve the stepsize and start over.
	Ui = Energy.TotalEnergy(adatoms, substrate_bins)
	N = len(adatoms)
	xn = adatoms[:]
	xnp = []
	# Initial forces on each atom
	dx0 = [Energy.AdatomAdatomForce(i, xn) + Energy.AdatomSurfaceForce(xn[i], substrate_bins) for i in range(N)]
	for i in range(N):
		other_adatoms = xn[:]
		other_adatoms.pop(i)
		# Search for the lowest energy along the force vector
		xs = [(xn[i] + a*dx0[i]/10/scale, other_adatoms, substrate_bins) for a in range(30)]
		ys = pool.map(Energy.RelaxEnergy, xs)
		(xmin, a, a) = xs[ys.index(min(ys))]
		xs = [(xmin - dx0[i]/10/scale + a*dx0[0]/50/scale, other_adatoms, substrate_bins) for a in range(20)]
		ys = pool.map(Energy.RelaxEnergy, xs)
		(xmin, a, a) = xs[ys.index(min(ys))]
		xnp.append(xmin)
	sn = dx0[:]
	snp = dx0[:]
	# Force on each atom in the lowest energy position
	dxnp = [Energy.AdatomAdatomForce(i, xnp) + Energy.AdatomSurfaceForce(xnp[i], substrate_bins) for i in range(N)]
	maxF = max([np.dot(dxi, dxi) for dxi in dxnp])
	lastmaxF = maxF
	while maxF > threshold:
		for i in range(N):
			# Calculate the conjugate direction step size, beta
			top = np.dot(xnp[i], (xnp[i] - xn[i]))
			bot = np.dot(xn[i], xn[i])
			beta = top/bot
			# Update conjugate direction
			snp[i] = dxnp[i] + beta*sn[i]
			
		xn = xnp[:]
		sn = snp[:]
		
		for i in range(N):
			other_adatoms = xn[:]
			other_adatoms.pop(i)
			# Calculate lowester energy position along the conjugate direction vector
			xs = [(xn[i] + a*sn[i]/10/scale, other_adatoms, substrate_bins) for a in range(30)]
			ys = pool.map(Energy.RelaxEnergy, xs)
			(xmin, a, a) = xs[ys.index(min(ys))]
			xs = [(xmin - sn[i]/10/scale + a*sn[0]/50/scale, other_adatoms, substrate_bins) for a in range(20)]
			ys = pool.map(Energy.RelaxEnergy, xs)
			(xmin, a, a) = xs[ys.index(min(ys))]
			xmin = Periodic.PutInBox(xmin)
			xnp[i] = xmin
			if xmin[1] < -0.1 or xmin[1] > max([p[1] for p in backup])+gv.r_a*1.2:
				# Halve the step size if things went wonky
				print 'changing scale', scale*2, xmin
				return Relaxation(backup, substrate_bins, scale=scale*2, threshold=threshold)
		
		xn = xnp[:]
		# Calculate forces at new lowest energy position
		dxnp = [Energy.AdatomAdatomForce(i, xn) + Energy.AdatomSurfaceForce(xn[i], substrate_bins) for i in range(N)]
		maxF = max([np.dot(dxi, dxi) for dxi in dxnp])
		if abs(lastmaxF - maxF) < 1e-6 or maxF > 1e4 or Energy.TotalEnergy(adatoms, substrate_bins) > Ui:
			# Halve the step size if things went wonky
			print 'changing scale', scale*2, maxF
			return Relaxation(backup, substrate_bins, scale=scale*2, threshold=threshold)
		lastmaxF = maxF
	return xn

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
	nearby_adatoms = Bins.NearbyAtoms(around, adatom_bins)
	truth = zip(*[iter(np.in1d(adatoms, nearby_adatoms))]*2)
	truth = [True if t == (True, True) else False for t in truth]
	nearby_indices = []
	for i in range(len(adatoms)):
		if truth[i]:
			nearby_indices.append(i)
	nearby_indices = sorted(nearby_indices, reverse=True)
	for i in nearby_indices:
		relaxing_adatoms.append(adatoms.pop(i))
	relaxing_adatoms = Relaxation(relaxing_adatoms, substrate_bins, scale=16)
	adatoms += relaxing_adatoms
	forces = [Energy.AdatomAdatomForce(i, adatoms) + Energy.AdatomSurfaceForce(adatoms[i], substrate_bins) for i in range(len(adatoms))]
	maxF = max([np.dot(f, f) for f in forces])
	if maxF > 1e-1:
		print 'global required'
		adatoms = Relaxation(adatoms, substrate_bins, scale=16, threshold=1e-2)
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
	for a in atoms:
		plt.scatter(a[0], a[1], color=color)

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

adatoms = []
adatoms = InitSubstrate()[:gv.W-1]
adatoms = [adatoms[i] + np.array([gv.r_a/2, gv.r_a*gv.sqrt3/2]) for i in range(len(adatoms)-1)]
adatoms = Relaxation(adatoms, substrate_bins, scale=16)
t = 0
while True:
	Rk = HoppingRates(adatoms, substrate_bins)
	pj = HoppingPartialSums(Rk)
	Rd = DepositionRate()
	Rtot = TotalRate(Rd, Rk)
	r = random.random()*Rtot
	print t, pj[-1], Rd, r
	HoppingAtom = [p for p in pj if p > r]
	if len(HoppingAtom) > 0:
		print 'HOP'
		adatoms = HopAtom(pj.index(HoppingAtom[0])-1, adatoms, substrate_bins)
	else:
		adatoms = DepositAdatom(adatoms, substrate_bins)
	
	# if i % 5 == 0:
	PlotSubstrate(substrate, 'blue')
	PlotAtoms(adatoms, 'green')
	surf = SurfaceAtoms(adatoms, substrate_bins)
	PlotAtoms(surf, 'red')
	plt.axis([-gv.L/2-0.1, gv.L/2+0.1, gv.Dmin-0.1, gv.Dmax+3*gv.r_as])
	plt.savefig(gv.folder + '%.2i.png'%t)
	# plt.show()
	plt.clf()
	# PlotEnergy(adatoms, substrate_bins)
	t += 1

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