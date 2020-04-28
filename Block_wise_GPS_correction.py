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
from heapq import heappush, heappop, heapify

PATCH_SIZE = (3296, 2472)
PATCH_SIZE_GPS = (8.899999997424857e-06,1.0199999998405929e-05)
HEIGHT_RATIO_FOR_ROW_SEPARATION = 0.1
NUMBER_OF_ROWS_IN_GROUPS = 10

class GPS_Coordinate:
	
	def __init__(self,UL_coord,UR_coord,LL_coord,LR_coord,Center):
		self.UL_coord = UL_coord
		self.UR_coord = UR_coord
		self.LL_coord = LL_coord
		self.LR_coord = LR_coord
		self.Center = Center



class Patch:
	
	def __init__(self,name,coord):
		self.name = name
		self.gps = coord


class Group:
	def __init__(self,rows):
		self.rows = rows



class Field:
	def __init__(self,coordinates_file):
		self.groups = self.initialize_field(coordinates_file)

	def initialize_field(self,coordinates_file):

		rows = self.get_rows(coordinates_file)

		groups = []

		while len(groups)*NUMBER_OF_ROWS_IN_GROUPS<len(rows):
			
			iterator = len(groups)
			row_window = rows[iterator*NUMBER_OF_ROWS_IN_GROUPS:(iterator+1)*NUMBER_OF_ROWS_IN_GROUPS]
			group = Group(row_window)
			groups.append(group)

		return groups

	def get_rows(self,coordinates_file):

		center_of_rows = []
		patches = []
		

		with open(coordinates_file) as f:
			lines = f.read()
			lines = lines.replace('"','')

			for l in lines.split('\n'):
				if l == '':
					break
				if l == 'Filename,Upper left,Lower left,Upper right,Lower right,Center' or l == 'name,upperleft,lowerleft,uperright,lowerright,center':
					continue

				features = l.split(',')

				filename = features[0]
				upper_left = (float(features[1]),float(features[2]))
				lower_left = (float(features[3]),float(features[4]))
				upper_right = (float(features[5]),float(features[6]))
				lower_right = (float(features[7]),float(features[8]))
				center = (float(features[9]),float(features[10]))

				coord = GPS_Coordinate(upper_left,upper_right,lower_left,lower_right,center)
				patches.append(Patch(filename,coord))

				is_new = True

				for c in center_of_rows:
					if abs(center[1]-c[1]) < PATCH_SIZE_GPS[1]/HEIGHT_RATIO_FOR_ROW_SEPARATION:
						is_new = False

				if is_new:
					center_of_rows.append(center)

		patches_groups_by_rows = {}

		center_of_rows = sorted(center_of_rows, key=lambda x: x[1])

		for c in center_of_rows:
			patches_groups_by_rows[c] = []

		for p in patches:
			min_distance = PATCH_SIZE_GPS[1]*2
			min_row = None

			for c in center_of_rows:
				distance = abs(p.gps.Center[1]-c[1])
				if distance<min_distance:
					min_distance = distance
					min_row = c

			patches_groups_by_rows[min_row].append(p)


		rows = []
		
		for g in patches_groups_by_rows:
			newlist = sorted(patches_groups_by_rows[g], key=lambda x: x.gps.Center[0], reverse=False)
			
			rows.append(newlist)

		return rows

	def save_plot(self,path_to_save):
		result = []
		color = 0

		for group in self.groups:
			
			if color == 0:
				color = 1
			else:
				color = 0

			for row in g.rows:
				for p in row:
					result.append([p.gps.Center[0],p.gps.Center[1],color])
		
		np.save(path_to_save,np.array(result))	

def visualize_plot(path):
	import matplotlib.pyplot as plt

	plt.axis('equal')

	data = np.load(path)

	c = []
	for d in data:
		c.append('red' if d[2] == 1 else 'green')

	plt.scatter(data[:,0],data[:,1],color=c)

	plt.show()

def main():
	global server

	if server == 'coge':
		patch_folder = '/storage/ariyanzarei/2020-01-08-rgb/bin2tif_out'
		SIFT_folder = '/storage/ariyanzarei/2020-01-08-rgb/SIFT'
		lid_file = '/storage/ariyanzarei/2020-01-08-rgb/lids.txt'
		coordinates_file = '/storage/ariyanzarei/2020-01-08-rgb/2020-01-08_coordinates.csv'
		CORRECTED_coordinates_file = '/storage/ariyanzarei/2020-01-08-rgb/2020-01-08_coordinates_CORRECTED.csv'
		plot_npy_file = '/storage/ariyanzarei/2020-01-08-rgb/plt.npy'
		row_save_path = '/storage/ariyanzarei/2020-01-08-rgb/rows'

	elif server == 'laplace.cs.arizona.edu':
		patch_folder = '/data/plant/full_scans/2020-01-08-rgb/bin2tif_out'
		SIFT_folder = '/data/plant/full_scans/2020-01-08-rgb/SIFT'
		lid_file = '/data/plant/full_scans/2020-01-08-rgb/lids.txt'
		coordinates_file = '/data/plant/full_scans/metadata/2020-01-08_coordinates.csv'
		CORRECTED_coordinates_file = '/data/plant/full_scans/metadata/2020-01-08_coordinates_CORRECTED.csv'
		plot_npy_file = '/data/plant/full_scans/2020-01-08-rgb/plt.npy'

	elif server == 'ariyan':
		patch_folder = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/Figures'
		SIFT_folder = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/SIFT'
		lid_file = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/lids.txt'
		coordinates_file = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/coords.txt'
		CORRECTED_coordinates_file = '/home/ariyan/Desktop/200203_Mosaic_Training_Data/200203_Mosaic_Training_Data/coords2.txt'
		plot_npy_file = '/home/ariyan/Desktop/plt.npy'


	if server == 'coge':
		print('RUNNING ON -- {0} --'.format(server))
		field = Field(coordinates_file)
		field.save_plot(plot_npy_file)


	elif server == 'laplace.cs.arizona.edu':
		print('RUNNING ON -- {0} --'.format(server))
		os.system("taskset -p -c 1,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,39,44,45,46 %d" % os.getpid())


	elif server == 'ariyan':
		print('RUNNING ON -- {0} --'.format(server))
		visualize_plot(plot_npy_file)


def report_time(start,end):
	print('-----------------------------------------------------------')
	print('Start date time: {0}\nEnd date time: {1}\nTotal running time: {2}.'.format(start,end,end-start))

server_core = {'coge':64,'laplace.cs.arizona.edu':20,'ariyan':4}

server = socket.gethostname()
no_of_cores_to_use = server_core[server]

start_time = datetime.datetime.now()
main()
end_time = datetime.datetime.now()
report_time(start_time,end_time)
