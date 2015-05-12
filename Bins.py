#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np

import Constants as gv
import Periodic

def BinIndex(x, y):
	'''
	Position of an atom in 'bin-space'.
	
	Ri:       np.array[x, y] - position of atom
	L:        Float          - global variable indicating width of the periodic cell
	bin_size: Float          - global variable indicating width of a bin
	Dmin:     Float          - global variable indicating the lowest substrate atom y-position
	
	returns: (Int, Int) - indices of the atom in 'bin-space'
	'''
	nx = int((x + gv.L/2)/gv.bin_size)
	ny = int((y - gv.Dmin)/gv.bin_size)
	return (nx, ny)

def PutInBins(R):
	'''
	Which atoms go in which bins?
	
	R: np.array(np.array[x, y]) - list of atom positions
	
	returns: [[[np.array[x, y]]]] - bins of atom positions ordered column-first ([x][y])
	'''
	bins = []
	for i in range(len(R)/2):
		x = R[2*i]; y = R[2*i+1]
		(nx, ny) = BinIndex(x, y)
		while len(bins) <= nx:
			bins.append([])
		while len(bins[nx]) <= ny:
			bins[nx].append([])
		bins[nx][ny].append(x)
		bins[nx][ny].append(y)
	return np.array(bins)

def NearbyBins(x, y):
	'''
	Which bins are adjacent to this atom?
	
	Ri:      np.array[x, y] - position of atom
	nbins_x: Int            - global variable indicating number of bins along x-axis
	nbins_y: Int            - global variable indicating number of bins along y-axis
	
	returns: ([Int], [Int]) - indices of nearby bins separated by coordinate ([all-xs], [all-ys])
	'''
	(nx, ny) = BinIndex(x, y)
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
	if nx == 0:
		nearby_x.append(gv.nbins_x-2)
	if nx == gv.nbins_x-2:
		nearby_x.append(0)
	return (nearby_x, nearby_y)

def NearbyAtoms(x, y, R_bins):
	'''
	Find nearest atoms to Ri in R_bins.
	'''
	(nearby_x, nearby_y) = NearbyBins(x, y)
	nearby_atoms = []
	for x in nearby_x:
		for y in nearby_y:
			try:
				nearby_atoms += R_bins[x][y]
			except IndexError:
				pass # Bins with no atoms are not returned from PutInBins()
	return np.array(nearby_atoms)

def NearestNeighbors(adatoms, substrate_bins):
	'''
	Find closest adatoms and substrate atoms. Mostly used for counting bonds.
	'''
	nearest_adatoms, nearest_substrate = [], []
	adatom_bins = PutInBins(adatoms)
	for i in range(len(adatoms)/2):
		x = adatoms[2*i]; y = adatoms[2*i+1]
		nearby_adatoms = NearbyAtoms(x, y, adatom_bins)
		nearby_substrate = NearbyAtoms(x, y, substrate_bins)
		na_i, ns_i = [], []
		
		d = Periodic.Distances(np.array([x, y]), nearby_adatoms)[0]
		for j in range(len(d)):
			if d[j] < 1.2*gv.r_a:
				na_i.append(nearby_adatoms[2*j])
				na_i.append(nearby_adatoms[2*j+1])
		nearest_adatoms.append(np.array(na_i))
		
		d = Periodic.Distances(np.array([x, y]), nearby_substrate)[0]
		for j in range(len(d)):
			if d[j] < 1.2*gv.r_as:
				ns_i.append(nearby_substrate[2*j])
				ns_i.append(nearby_substrate[2*j+1])
		nearest_substrate.append(np.array(ns_i))
	return (np.array(nearest_adatoms), np.array(nearest_substrate))