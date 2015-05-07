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

def AdatomSurfaceSubsetEnergy(adatom, substrate_subset, E_as, r_as):
	return sum([PairwisePotential(adatom, Rs, E_as, r_as) for Rs in substrate_subset])

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
	return AdatomSurfaceSubsetEnergy(adatom, nearby_substrate, E_as, r_as)

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

def U_appx(i, adatoms, substrate_bins, E_as, r_as, E_a, r_a):
	Ri = adatoms[i]
	Uappx = AdatomSurfaceEnergy(Ri, substrate_bins, E_as, r_as) + AdatomAdatomEnergy(i, adatoms, E_a, r_a)
	return Uappx

def U_loc(i, adatoms, substrate_bins, E_as, r_as, E_a, r_a):
	Ri = adatoms[i]
	(nearest_adatoms, nearest_substrate) = Bins.NearestNeighbors(adatoms, substrate_bins, r_as, r_a)
	nearest_adatoms = nearest_adatoms[i]
	nearest_substrate = nearest_substrate[i]
	return AdatomAdatomEnergy(0, [Ri] + nearest_adatoms, E_a, r_a) + AdatomSurfaceSubsetEnergy(Ri, nearest_substrate, E_as, r_as)

def U_loc_ideal(i, adatoms, substrate_bins, E_as, r_as, E_a, r_a):
	Ri = adatoms[i]
	(nearest_adatoms, nearest_substrate) = Bins.NearestNeighbors(adatoms, substrate_bins, r_as, r_a)
	nearest_adatoms = nearest_adatoms[i]
	nearest_substrate = nearest_substrate[i]
	return -(len(nearest_adatoms) + len(nearest_substrate))*E_as

def C():
	return 1.0

def DeltaU(i, adatoms, substrate_bins, E_as, r_as, E_a, r_a):
	dU_appx = U_appx(i, adatoms, substrate_bins, E_as, r_as, E_a, r_a)
	dU_loc = U_loc(i, adatoms, substrate_bins, E_as, r_as, E_a, r_a)
	dU_loc_ideal = U_loc_ideal(i, adatoms, substrate_bins, E_as, r_as, E_a, r_a)
	return C()*(dU_loc - dU_loc_ideal) + dU_appx