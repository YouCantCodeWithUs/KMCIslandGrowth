#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np

import Constants as gv
import Periodic
import Bins

def PairwisePotential(Ri, Rj, E_ij, r_ij):
	'''
	Lennard-Jones potential between two atoms.
	
	Ri:   np.array[x, y] - position of first atom
	Rj:   np.array[x, y] - position of second atom
	E_ij: Float          - binding energy between Ri and Rj
	r_ij: Float          - equilibrium bond length between Ri and Rj
	
	returns: Float - potential energy of Ri due to Rj (will return 0 if Ri == Rj)
	'''
	r = Periodic.Distance(Ri, Rj)
	if r == 0:
		return 0
	return 4*E_ij*((r_ij/r)**12 - (r_ij/r)**6)

def PairwiseForce(Ri, Rj, E_ij, r_ij):
	r_vec = Periodic.Displacement(Ri, Rj)
	r_mag = np.sqrt(np.dot(r_vec, r_vec))
	if r_mag == 0:
		return np.zeros(3)
	f = 4*E_ij*(-12*r_ij**12/r_mag**13 + 6*r_ij**6/r_mag**7)
	return r_vec/r_mag*f

def AdatomSurfaceEnergy(adatom, substrate_bins, E_as, r_as):
	'''
	Lennard-Jones potential between an adatom and the substrate atoms.
	
	adatom:         np.array[x, y]       - position of adatom
	substrate_bins: [[[np.array[x, y]]]] - binned substrate positions
	E_as:           Float                - binding energy between adatoms and substrate atoms
	r_as:           Float                - equilibrium bond length between adatoms and substrate atoms
	
	returns: Float - potential energy of adatom due to substrate atoms
	'''
	nearby_substrate = Bins.NearbyAtoms(adatom, substrate_bins)
	return sum([PairwisePotential(adatom, Rs, E_as, r_as) for Rs in nearby_substrate])

def AdatomAdatomEnergy(i, adatoms, E_a, r_a):
	'''
	Lennard-Jones potential between an adatom and other adatoms.
	
	i:       Int                      - index of adatom
	adatoms: np.array(np.array[x, y]) - position of adatom
	E_a:     Float                    - binding energy between adatoms and substrate atoms
	r_a:     Float                    - equilibrium bond length between adatoms and substrate atoms
	
	returns: Float - potential energy of an adatom due to other adatoms
	'''
	Ri = adatoms[i]
	# faster for small systems, slower for big systems
	if len(adatoms) < 1000:
		return sum([PairwisePotential(Ri, Rj, E_a, r_a) for Rj in adatoms])
	nearby_adatoms = Bins.NearbyAtoms(Ri, adatoms)
	return sum([PairwisePotential(Ri, Rj, E_a, r_a) for Rj in nearby_adatoms])

def AdatomSurfaceForce(adatom, substrate_bins, E_as, r_as):
	nearby_substrate = Bins.NearbyAtoms(adatom, substrate_bins)
	return sum([PairwiseForce(adatom, Rs, E_as, r_as) for Rs in nearby_substrate])

def AdatomAdatomForce(i, adatoms, E_a, r_a):
	Ri = adatoms[i]
	if len(adatoms) < 1000:
		return sum([PairwiseForce(Ri, Rj, E_a, r_a) for Rj in adatoms])
	nearby_adatoms = Bins.NearbyAtoms(Ri, adatoms)
	return sum([PairwiseForce(Ri, Rj, E_a, r_a) for Rj in nearby_adatoms])