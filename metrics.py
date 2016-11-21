#!/usr/bin/python2

import numpy as np
from scipy.stats.mstats import gmean


# http://homepage.univie.ac.at/robert.kunst/prognos7.pdf


def _ape(y_true, y_pred):
	return 100. * np.absolute((y_true - y_pred) / y_true)

def mape(y_true, y_pred):
	# The MAPE (Mean Absolute Percent Error) measures the size of
	# the error in percentage terms. It is calculated as the average of
	# the unsigned percentage error
	return np.mean(_ape(y_true, y_pred))

mapd = mape

def mdape(y_true, y_pred):
	return np.median(_ape(y_true, y_pred))

def mse(y_true, y_pred):
	return ((y_true - y_pred) ** 2).mean()

def mae(y_true, y_pred):
	return np.mean(np.absolute(y_true - y_pred))

mad = mae

def rmse(y_true, y_pred):
	return np.sqrt(mse(y_true, y_pred))

def _rae(y_true, y_pred, y_benchmark_pred):
	# http://www.gepsoft.com/gxpt4kb/Chapter10/Section2/SS15.htm - bullshit alert!
	return (np.absolute(y_pred - y_true) / np.absolute(y_benchmark_pred - y_true))

def mdrae(y_true, y_pred):
	# warning! 
	assert len(y_pred) == len(y_true) - 1, "Do not predict the first value!"
	# benchmark - always predict the previous one
	y_benchmark_pred = y_true[:-1]
	return np.median(rae(y_true, y_pred, y_benchmark_pred))

def gmrae(y_true, y_pred):
	# warning! 
	assert len(y_pred) == len(y_true) - 1, "Do not predict the first value!"
	# benchmark - always predict the previous one
	y_benchmark_pred = y_true[:-1]
	return gmean(rae(y_true, y_pred, y_benchmark_pred))

def mase(y_true, y_pred, m=1):
	# https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
	return mae(y_true, y_pred) / mae(y_true[m:], y_true[:-m])

def gmae(y_true, y_pred):
	return gmean(np.absolute(y_true - y_pred))

def smape(y_true, y_pred):
	# https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
	# sometimes 200. = 100.
	return np.mean(200. * np.absolute(y_true - y_pred) / (np.absolute(y_true) + np.absolute(y_pred)))

def percent_better(y_true, y_pred1, y_pred2):
	return float((np.absolute(y_true - y_pred1) >= np.absolute(y_true - y_pred2)).sum()) / float(len(y_true))



