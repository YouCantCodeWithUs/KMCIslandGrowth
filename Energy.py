#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
from multiprocessing import Pool

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
	return E_ij*((r_ij/r)**12 - 2*(r_ij/r)**6)

def PairwiseForce(Ri, Rj, E_ij, r_ij):
	r_vec = Periodic.Displacement(Ri, Rj)
	r_mag = np.sqrt(np.dot(r_vec, r_vec))
	if r_mag == 0:
		return np.zeros(2)
	f = E_ij*(-12*r_ij**12/r_mag**13 + 12*r_ij**6/r_mag**7)
	return r_vec/r_mag*f

def AdatomSurfaceSubsetEnergy(adatom, substrate_subset):
	return sum([PairwisePotential(adatom, Rs, gv.E_as, gv.r_as) for Rs in substrate_subset])

def AdatomSurfaceEnergy(adatom, substrate_bins):
	'''
	Lennard-Jones potential between an adatom and the substrate atoms.
	
	adatom:         np.array[x, y]       - position of adatom
	substrate_bins: [[[np.array[x, y]]]] - binned substrate positions
	gv.E_as:           Float                - binding energy between adatoms and substrate atoms
	gv.r_as:           Float                - equilibrium bond length between adatoms and substrate atoms
	
	returns: Float - potential energy of adatom due to substrate atoms
	'''
	nearby_substrate = Bins.NearbyAtoms(adatom, substrate_bins)
	return AdatomSurfaceSubsetEnergy(adatom, nearby_substrate)

def AdatomAdatomEnergy(i, adatoms):
	'''
	Lennard-Jones potential between an adatom and other adatoms.
	
	i:       Int                      - index of adatom
	adatoms: np.array(np.array[x, y]) - position of adatom
	gv.E_a:     Float                    - binding energy between adatoms and substrate atoms
	gv.r_a:     Float                    - equilibrium bond length between adatoms and substrate atoms
	
	returns: Float - potential energy of an adatom due to other adatoms
	'''
	Ri = adatoms[i]
	return ArbitraryAdatomEnergy(Ri, adatoms)

def ArbitraryAdatomEnergy(Ri, adatoms):
	# faster for small systems, slower for big systems
	if len(adatoms) < 1000:
		return sum([PairwisePotential(Ri, Rj, gv.E_a, gv.r_a) for Rj in adatoms])
	nearby_adatoms = Bins.NearbyAtoms(Ri, adatoms)
	return sum([PairwisePotential(Ri, Rj, gv.E_a, gv.r_a) for Rj in nearby_adatoms])

def RelaxEnergy(args):
	(x, other_adatoms, substrate_bins) = args
	return ArbitraryAdatomEnergy(x, other_adatoms) + AdatomSurfaceEnergy(x, substrate_bins)

def AdatomSurfaceForce(adatom, substrate_bins):
	nearby_substrate = Bins.NearbyAtoms(adatom, substrate_bins)
	return AdatomSurfaceSubsetForce(adatom, nearby_substrate)

def AdatomSurfaceSubsetForce(adatom, substrate_subset):
	return sum([PairwiseForce(adatom, Rs, gv.E_as, gv.r_as) for Rs in substrate_subset])

def AdatomAdatomForce(i, adatoms):
	Ri = adatoms[i]
	return ArbitraryAdatomForce(Ri, adatoms)

def ArbitraryAdatomForce(Ri, adatoms):
	if len(adatoms) < 1000:
		return sum([PairwiseForce(Ri, Rj, gv.E_a, gv.r_a) for Rj in adatoms])
	nearby_adatoms = Bins.NearbyAtoms(Ri, adatoms)
	return sum([PairwiseForce(Ri, Rj, gv.E_a, gv.r_a) for Rj in nearby_adatoms])

def U_appx(i, adatoms, substrate_bins):
	Ri = adatoms[i]
	Uappx = AdatomSurfaceEnergy(Ri, substrate_bins) + AdatomAdatomEnergy(i, adatoms)
	return Uappx

def U_loc(i, adatoms, substrate_bins):
	Ri = adatoms[i]
	(nearest_adatoms, nearest_substrate) = Bins.NearestNeighbors(adatoms, substrate_bins)
	nearest_adatoms = nearest_adatoms[i]
	nearest_substrate = nearest_substrate[i]
	return AdatomAdatomEnergy(0, [Ri] + nearest_adatoms) + AdatomSurfaceSubsetEnergy(Ri, nearest_substrate)

def U_loc_ideal(i, adatoms, substrate_bins):
	Ri = adatoms[i]
	(nearest_adatoms, nearest_substrate) = Bins.NearestNeighbors(adatoms, substrate_bins)
	nearest_adatoms = nearest_adatoms[i]
	nearest_substrate = nearest_substrate[i]
	return -(len(nearest_adatoms) + len(nearest_substrate))*gv.E_as

def C():
	return 1.0

def DeltaU(i, adatoms, substrate_bins):
	dU_appx = U_appx(i, adatoms, substrate_bins)
	dU_loc = U_loc(i, adatoms, substrate_bins)
	dU_loc_ideal = U_loc_ideal(i, adatoms, substrate_bins)
	return C()*(dU_loc - dU_loc_ideal) + dU_appx