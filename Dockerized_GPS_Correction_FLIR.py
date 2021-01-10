import argparse
import numpy as np
import cv2
import random
import math
import multiprocessing
import datetime
import sys
import gc
import pickle
import os
import threading
import socket
import statistics
import datetime
from Customized_myltiprocessing import MyPool
from heapq import heappush, heappop, heapify
from collections import OrderedDict,Counter

from Block_wise_GPS_correction import *
import settings

def main(scan_date):

	sys.setrecursionlimit(10**8)

	field = Field(is_single_group=settings.is_single_group)
	
	field.save_plot()
	
	old_RMSE = get_approximate_random_RMSE_overlap(field,10,settings.no_of_cores_to_use_max)

	field.create_patches_SIFT_files()
	
	field.draw_and_save_field(is_old=True)

	field.correct_field()

	field.save_new_coordinate()

	field.draw_and_save_field(is_old=False)

	new_RMSE = get_approximate_random_RMSE_overlap(field,10,settings.no_of_cores_to_use_max)

	print('------------------ ERROR MEASUREMENT ------------------ ')

	print('OLD SI: {0}'.format(np.mean(old_RMSE[:,3])))
	
	print('NEW SI: {0}'.format(np.mean(new_RMSE[:,3])))


def get_args():

	parser = argparse.ArgumentParser(description='Geo-correction FLIR images on HPC.')
	parser.add_argument('-d','--destination', type=str, help='address of the destination folder on HPC (usually on xdisk where everything is stored).')
	parser.add_argument('-b','--bin_2tif', type=str, help='the address of the bin_2tif folder (or any folder that contains all the images).')
	parser.add_argument('-s','--scan_date', type=str, help='the name of the specific scan to work on.')
	parser.add_argument('-c','--config_file', type=str, help='the name of the config file to use.')
	parser.add_argument('-l','--lid_address', type=str, help='the address of the lid file.')
	parser.add_argument('-r','--repository_address', type=str, help='the address of the geocorrection repository.')

	args = parser.parse_args()

	return args

start_time = datetime.datetime.now()

args = get_args()

scan_date = args.scan_date
config_file = args.config_file
destination = args.destination
lid_file_address = args.lid_address
bin2tiff_address = args.bin_2tif
repository_address = args.repository_address

if not os.path.exists('{0}/{1}'.format(destination,scan_date)):
	os.mkdir('{0}/{1}'.format(destination,scan_date))
if not os.path.exists('{0}/{1}/SIFT'.format(destination,scan_date)):
	os.mkdir('{0}/{1}/SIFT'.format(destination,scan_date))
if not os.path.exists('{0}/{1}/logs'.format(destination,scan_date)):
	os.mkdir('{0}/{1}/logs'.format(destination,scan_date))

print('Geo-correction started. Log is being saved in {0}'.format(destination))

original = sys.stdout

sys.stdout = open('{0}/{1}/{2}.txt'.format(destination,scan_date,'geo_correction_output'), 'w')

settings.initialize_settings_HPC(scan_date,config_file,destination,lid_file_address,bin2tiff_address,repository_address)

print_settings()

main(scan_date)

end_time = datetime.datetime.now()

report_time(start_time,end_time)

sys.stdout = original

