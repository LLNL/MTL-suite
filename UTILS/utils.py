"""Generic object pickler and compressor

This module saves and reloads compressed representations of generic Python
objects to and from the disk.
"""

__author__ = "Bill McNeill <billmcn@speakeasy.net>"
__version__ = "1.0"

import pickle
import gzip


def save(object, filename, bin = 1):
	"""Saves a compressed object to disk
	"""
	file = gzip.GzipFile(filename, 'wb')
	file.write(pickle.dumps(object, bin))
	file.close()


def load(filename):
	"""Loads a compressed object from disk
	"""
	file = gzip.GzipFile(filename, 'rb')
	buffer = ""
	while 1:
		data = file.read()
		if data == "":
			break
		buffer += data
	object = pickle.loads(buffer)
	file.close()
	return object

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()