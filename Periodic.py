#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np

import Constants as gv

def PutInBox(Ri):
	'''
	Implement periodic boundary conditions along the x-axis.
	
	Ri: np.array[x, y] - position of a single atom
	L:  Float          - global variable indicating the width of the periodic cell
	
	returns: np.array[x, y] - correct position using PBC
	'''
	x = Ri[0] % gv.L
	if abs(x) > gv.L/2:
		x += gv.L if gv.L < 0 else -gv.L
	return np.array([x, Ri[1]])

def Displacement(Ri, Rj):
	'''
	Least-distance spacial displacement between two atoms.
	
	Ri: np.array[x, y] - position of first atom
	Rj: np.array[x, y] - position of second atom
	
	returns: np.array[x, y] - vector pointing from Ri to Rj
	'''
	return PutInBox(Rj-Ri)

def Distance(Ri, Rj):
	'''
	Least-distance between two atoms.
	
	Ri: np.array[x, y] - position of first atom
	Rj: np.array[x, y] - position of second atom
	
	returns: Float - distance between Ri and Rj
	'''
	d = Displacement(Ri, Rj)
	return np.sqrt(np.dot(d, d))