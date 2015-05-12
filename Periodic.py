#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np

import Constants as gv

def PutInBox(x):
	'''
	Implement periodic boundary conditions along the x-axis.
	
	Ri: np.array[x, y] - position of a single atom
	L:  Float          - global variable indicating the width of the periodic cell
	
	returns: np.array[x, y] - correct position using PBC
	'''
	x = x % gv.L
	if abs(x) > gv.L/2:
		x += gv.L if gv.L < 0 else -gv.L
	return x

def PutAllInBox(R):
	for i in range(len(R)/2):
		R[2*i] = PutInBox(R[2*i])
	return R

def Displacement(Ri, Rj):
	'''
	Least-distance spacial displacement between two atoms.
	
	Ri: np.array[x, y] - position of first atom
	Rj: np.array[x, y] - position of second atom
	
	returns: np.array[x, y] - vector pointing from Ri to Rj
	'''
	return PutAllInBox(Rj-Ri)

def Distance(Ri, Rj):
	'''
	Least-distance between two atoms.
	
	Ri: np.array[x, y] - position of first atom
	Rj: np.array[x, y] - position of second atom
	
	returns: Float - distance between Ri and Rj
	'''
	d = Displacement(Ri, Rj)
	return np.sqrt(np.dot(d, d))

def Displacements(x, y, Rjs):
	d = np.array([x, y]*(len(Rjs)/2))
	return Displacement(d, Rjs)

def Distances(a, b):
	Ds = []
	for i in range(len(a)/2):
		x = a[2*i]; y = a[2*i+1]
		d = Displacements(x, y, b)
		d *= d
		D = [np.sqrt(d[2*j] + d[2*j+1]) for j in range(len(d)/2)]
		Ds.append(np.array(D))
	return np.array(Ds)