#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np

import Constants as gv
import Periodic

def BinIndex(Ri):
	'''
	Position of an atom in 'bin-space'.
	
	Ri:       np.array[x, y] - position of atom
	L:        Float          - global variable indicating width of the periodic cell
	bin_size: Float          - global variable indicating width of a bin
	Dmin:     Float          - global variable indicating the lowest substrate atom y-position
	
	returns: (Int, Int) - indices of the atom in 'bin-space'
	'''
	nx = int((Ri[0] + gv.L/2)/gv.bin_size)
	ny = int((Ri[1] - gv.Dmin)/gv.bin_size)
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
			nearby_x[i] += gv.nbins_x
		if nearby_x[i] >= gv.nbins_x:
			nearby_x[i] -= gv.nbins_x
		if nearby_y[i] < 0:
			nearby_y[i] += gv.nbins_y
		if nearby_y[i] >= gv.nbins_y:
			nearby_y[i] -= gv.nbins_y
	return (nearby_x, nearby_y)

def NearbyAtoms(Ri, R_bins):
	(nearby_x, nearby_y) = NearbyBins(Ri)
	nearby_atoms = []
	for x in nearby_x:
		for y in nearby_y:
			try:
				nearby_atoms += R_bins[x][y]
			except IndexError:
				pass # Bins with no atoms are not returned from PutInBins()
	return nearby_atoms

def NearestNeighbors(adatoms, substrate_bins, r_as, r_a):
	nearest_adatoms, nearest_substrate = [], []
	adatom_bins = PutInBins(adatoms)
	for Ri in adatoms:
		nearby_adatoms = NearbyAtoms(Ri, adatom_bins)
		nearby_substrate = NearbyAtoms(Ri, substrate_bins)
		na_i, ns_i = [], []
		for Rj in nearby_adatoms:
			d = Periodic.Displacement(Ri, Rj)
			if abs(d[0]) < 1.2*r_a and abs(d[1]) < 1.2*r_a:
				d = np.sqrt(np.dot(d, d))
				if d == 0:
					pass
				elif d < 1.2*r_a:
					na_i.append(Rj)
		nearest_adatoms.append(na_i)
		for Rj in nearby_substrate:
			d = Periodic.Displacement(Ri, Rj)
			if abs(d[0]) < 1.2*r_as and abs(d[1]) < 1.2*r_as:
				d = np.sqrt(np.dot(d, d))
				ns_i.append(Rj)
		nearest_substrate.append(ns_i)
	return (nearest_adatoms, nearest_substrate)