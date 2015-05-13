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
	return E_ij*((r_ij/r)**12 - 2*(r_ij/r)**6)

def PairwisePotentials(Ri, Rjs, E_ij, r_ij):
	Ds = Periodic.Distances(Ri, Rjs)[0]
	Us = [E_ij*((r_ij/d)**12 - 2*(r_ij/d)**6) if d > 0 else 0 for d in Ds]
	return Us

def PairwiseForce(Ri, Rj, E_ij, r_ij):
	'''
	Lennard-Jones force between two atoms.
	
	Ri:   np.array[x, y] - position of first atom
	Rj:   np.array[x, y] - position of second atom
	E_ij: Float          - binding energy between Ri and Rj
	r_ij: Float          - equilibrium bond length between Ri and Rj
	
	returns: np.array[x, y] - force on Ri due to Rj (will return 0 if Ri == Rj)
	'''
	r_vec = Periodic.Displacement(Ri, Rj)
	r_mag = np.sqrt(np.dot(r_vec, r_vec))
	if r_mag == 0:
		return np.zeros(2)
	f = E_ij*(-12*r_ij**12/r_mag**13 + 12*r_ij**6/r_mag**7)
	return r_vec/r_mag*f

def PairwiseForces(Ri, Rjs, E_ij, r_ij):
	r_vecs = Periodic.Displacements(Ri[0], Ri[1], Rjs)
	r_mags = r_vecs*r_vecs
	r_mags = [np.sqrt(r_mags[2*j] + r_mags[2*j+1]) for j in range(len(r_mags)/2)]
	F = np.array([0., 0.])
	for i in range(len(r_mags)):
		if r_mags[i] < 1e-2:
			continue
		f = E_ij*(-12*r_ij**12/r_mags[i]**13 + 12*r_ij**6/r_mags[i]**7)
		F[0] += r_vecs[2*i]/r_mags[i]*f
		F[1] += r_vecs[2*i+1]/r_mags[i]*f
	return F

def AdatomAdatomForces(adatoms):
	Fs = []
	for i in range(len(adatoms)/2):
		F = PairwiseForces(adatoms[2*i:2*i+2], adatoms, gv.E_a, gv.r_a)
		Fs.append(F[0]); Fs.append(F[1])
	return np.array(Fs)

def AdatomSubstrateForces(adatoms, substrate_bins):
	Fs = []
	for i in range(len(adatoms)/2):
		nearby_substrate = Bins.NearbyAtoms(adatoms[2*i], adatoms[2*i+1], substrate_bins)
		F = PairwiseForces(adatoms[2*i:2*i+2], nearby_substrate, gv.E_as, gv.r_as)
		Fs.append(F[0]); Fs.append(F[1])
	return np.array(Fs)

def AdatomSurfaceSubsetEnergy(adatom, substrate_subset):
	'''
	Modular code! I think this is used somewhere other than AdatomSurfaceEnergy() for something. Maybe.
	'''
	return sum(PairwisePotentials(adatom, substrate_subset, gv.E_as, gv.r_as))

def AdatomSurfaceEnergy(adatom, substrate_bins):
	'''
	Lennard-Jones potential between an adatom and the substrate atoms.
	
	adatom:         np.array[x, y]       - position of adatom
	substrate_bins: [[[np.array[x, y]]]] - binned substrate positions
	gv.E_as:           Float                - binding energy between adatoms and substrate atoms
	gv.r_as:           Float                - equilibrium bond length between adatoms and substrate atoms
	
	returns: Float - potential energy of adatom due to substrate atoms
	'''
	nearby_substrate = Bins.NearbyAtoms(adatom[0], adatom[1], substrate_bins)
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
	'''
	Required for finding lowest energy position along force vectors.
	'''
	# faster for small systems, slower for big systems
	if len(adatoms) < 1000:
		return sum([PairwisePotential(Ri, Rj, gv.E_a, gv.r_a) for Rj in adatoms])
	nearby_adatoms = Bins.NearbyAtoms(Ri, adatoms)
	return sum([PairwisePotential(Ri, Rj, gv.E_a, gv.r_a) for Rj in nearby_adatoms])

def TotalEnergy(adatoms, substrate_bins):
	'''
	Used to compare potentially relaxed states to the unrelaxed state.
	'''
	U = 0
	for i in range(len(adatoms)/2-2):
		Ri = np.array([adatoms[2*i], adatoms[2*i+1]])
		U += sum(PairwisePotentials(Ri, adatoms[2*i+2:], gv.E_a, gv.r_a))
		U += AdatomSurfaceEnergy(Ri, substrate_bins)
	U += AdatomSurfaceEnergy(adatoms[-2:], substrate_bins)
	return U

def RelaxEnergy(args):
	'''
	Function required for multiprocessing during relaxation.
	
	args: (np.array[x, y], list(np.array[x, y], list(list(np.array[x, y])))) - (proposed relaxed position, position of other adatoms, substrate bins)
	
	returns: float - energy at the proposed relaxed position
	'''
	(xn, substrate_bins) = args
	xn = [np.array([xn[2*i], xn[2*i+1]]) for i in range(len(xn)/2)]
	U = 0
	for i in range(len(xn)):
		Ri = xn.pop(0)
		U += ArbitraryAdatomEnergy(Ri, xn)
		U += AdatomSurfaceEnergy(Ri, substrate_bins)
	return U

def AdatomSurfaceForce(adatom, substrate_bins):
	'''
	Force between an adatom and the substrate.
	'''
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

def HoppingEnergy(pos, adatoms, substrate_bins):
	nearby_substrate = Bins.NearbyAtoms(pos[0], pos[1], substrate_bins)
	return sum(PairwisePotentials(pos, adatoms, gv.E_a, gv.r_a)) + sum(PairwisePotentials(pos, nearby_substrate, gv.E_as, gv.r_as))

def U_appx(i, adatoms, substrate_bins):
	'''
	Energy required to remove an adatom.
	'''
	
	Ri = adatoms[2*i:2*i+2]
	nearby_substrate = Bins.NearbyAtoms(Ri[0], Ri[1], substrate_bins)
	return sum(PairwisePotentials(Ri, adatoms, gv.E_a, gv.r_a)) + sum(PairwisePotentials(Ri, nearby_substrate, gv.E_as, gv.r_as))

def U_loc(i, adatoms, substrate_bins):
	'''
	Energy of an adatom's nearest neighbor bonds in situ.
	'''
	
	Ri = adatoms[2*i:2*i+2]
	(nearest_adatoms, nearest_substrate) = Bins.NearestNeighbors(adatoms, substrate_bins)
	nearest_adatoms = nearest_adatoms[i]
	nearest_substrate = nearest_substrate[i]
	return sum(PairwisePotentials(Ri, nearest_adatoms, gv.E_a, gv.r_a)) + sum(PairwisePotentials(Ri, nearest_substrate, gv.E_as, gv.r_as))

def U_loc_ideal(i, adatoms, substrate_bins):
	'''
	Energy of an adatom's nearest neighbor bonds in an ideal lattice.
	'''
	Ri = adatoms[2*i:2*i+2]
	(nearest_adatoms, nearest_substrate) = Bins.NearestNeighbors(adatoms, substrate_bins)
	nearest_adatoms = nearest_adatoms[i]
	nearest_substrate = nearest_substrate[i]
	return -(len(nearest_adatoms)/2*gv.E_a + len(nearest_substrate)/2*gv.E_as)

def C():
	'''
	Scaling factor for lattice misfit based on the Stoney equation
	'''
	return 1.0

def DeltaU(i, adatoms, substrate_bins):
	'''
	Bring it all together to calculate the difference between the in situ lattice and an ideal lattice.
	'''
	dU_appx = U_appx(i, adatoms, substrate_bins)
	dU_loc = U_loc(i, adatoms, substrate_bins)
	dU_loc_ideal = U_loc_ideal(i, adatoms, substrate_bins)
	return C()*(dU_loc - dU_loc_ideal) + dU_appx