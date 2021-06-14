""" 
The MIT License (MIT)
Copyright (c) 2018, Nicolas Coudray and Aristotelis Tsirigos (NYU)
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from os import replace
import os.path
import re
import sys
import pickle
import json
import csv
import numpy as np
import glob


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import scipy.misc
# from scipy.misc import imsave
# from scipy.misc import imread
from imageio import imwrite as imsave
from imageio import imread

FLAGS = None



def dict_tiles_stats(filepath):
	stats_dict = {}
	if filepath == '':
		return stats_dict
	with open(filepath) as f:
		for line in f:
			line2 = line.replace('[','').replace(']','').split()
			if len(line2)>0:	
				
				stats_dict['.'.join(line2[0].split('.')[:-1])] = line
	return stats_dict





def get_inference_from_file(TileName, stats_dict):
	if TileName in stats_dict.keys():
		line = stats_dict[TileName]
		lineProb = [prob.replace('\n', '') for prob in line.split(',')[-3:]]
		
		NumberOfClasses = len(lineProb)
		class_all = []
		sum_class = 0
		for nC in range(0,NumberOfClasses):
			class_all.append(float(lineProb[nC]))
			sum_class = sum_class + float(lineProb[nC])
		for nC in range(NumberOfClasses-1):
			if sum_class == 0:
				sum_class = 1
			class_all[nC] = class_all[nC] / sum_class
	else:
		print("image not found in text file %s ... and that's weird..." % TileName)
	return class_all

def main():
	nClasses = [int(x) for x in FLAGS.Classes.split(',')]
	print("nClasses: " + str(nClasses))

	if FLAGS.threshold == '':
		nThresh = ''
	else:
		nThresh = [float(x) for x in FLAGS.threshold.split(',')]
		if len(nThresh) != len(nClasses):
			sys.exit('the length of the threshold option must match the one of the class option')
	SlideRootName = ''
	SlideNames = []
	idx = [{},{},{}]
	iv1 = {}
	iv2 = {}
	iv3 = {}
	count_tiles = {}
	ListSlideNames = {}
	skip = False

	# Read out_filename stats:
	input_paths = FLAGS.tiles_stats.split(',')
	stats_dict = [{},{},{}]
	print(FLAGS.tiles_stats)
	print(input_paths)
	print(len(input_paths))
	if len(input_paths) == 1:
		stats_dict[0] = dict_tiles_stats(input_paths[0])
		stats_dict[1] = stats_dict[0]
		stats_dict[2] = stats_dict[0]
	elif len(input_paths) == 3:
		stats_dict[0] = dict_tiles_stats(input_paths[0])
		stats_dict[1] = dict_tiles_stats(input_paths[1])
		stats_dict[2] = dict_tiles_stats(input_paths[2])
	elif len(input_paths) == 2:
		if len(nClasses) != 2:
			sys.exit('the number of input files does not match the number of classes used (2)!')
		else:
			stats_dict[0] = dict_tiles_stats(input_paths[0])
			stats_dict[1] = dict_tiles_stats(input_paths[1])
			stats_dict[2] = {}
	else:
		sys.exit('the number of input files does not match the number of classes used (2)!')


	
	filtered_dict = [{},{},{}]
	for k in stats_dict[0].keys():
		#print(k)
		if FLAGS.slide_filter in k:
			filtered_dict[0][k] = stats_dict[0][k]
	for k in stats_dict[1].keys():
		#print(k)
		if FLAGS.slide_filter in k:
			filtered_dict[1][k] = stats_dict[1][k]
	for k in stats_dict[2].keys():
		if FLAGS.slide_filter in k:
			filtered_dict[2][k] = stats_dict[2][k]

	## Aggregate the results and build heatmaps
	Start = True
	NewSlide = True
	# For each image in the out_filename_stats:
	for cChannel in range(3):
		print("Channel " + str(cChannel) + ", " + str(len(filtered_dict[cChannel].keys())) + " images")
		tmp_count = 0
		for tile in sorted(filtered_dict[cChannel].keys()):
			tmp_count += 1
			print(str(tmp_count) + " iterations of  " + str(cChannel))
			cTileRootName =  '_'.join(tile.split('_')[0:-2]) 

			if cTileRootName == '':
				print("empty field")
				continue
			elif cTileRootName == SlideRootName:
				if skip:
					continue

				NewSlide = False
			else:
				if cTileRootName not in idx[0].keys() :
					idx[0][cTileRootName] = [(), ()]	
					idx[1][cTileRootName] = [(), ()]	
					idx[2][cTileRootName] = [(), ()]	
					iv1[cTileRootName] = []	
					iv2[cTileRootName] = []	
					iv3[cTileRootName] = []	
					ListSlideNames[cTileRootName] = []
					count_tiles[cTileRootName] = [0, 0, 0]

				NewSlide = True
				skip = False
			SlideRootName = cTileRootName
					#else:


			# extract coordinates of the tile
			ixTile = int(tile.split('_')[-2])
			iyTile = int(tile.split('_')[-1].split('.')[0])

			class_all = get_inference_from_file(tile, stats_dict[cChannel])

			if (len(nClasses) > 0) & (cChannel == 0) :
				if nClasses[0] > 0:
					if nThresh != '':
						newProb = class_all[nClasses[0]-1] / nThresh[cChannel] * 0.5
						if newProb > 0.5:
							newProb = (newProb - 0.5) / (1.0 / nThresh[cChannel] - 1) + 0.5 
						iv1[cTileRootName].append(newProb)
						if class_all[nClasses[cChannel]-1] > nThresh[cChannel]:
							count_tiles[cTileRootName][cChannel] = count_tiles[cTileRootName][cChannel] + 1
					else:
						iv1[cTileRootName].append(class_all[nClasses[0]-1])
						if class_all[nClasses[0]-1] == max(class_all):
							count_tiles[cTileRootName][0] = count_tiles[cTileRootName][0] + 1

				else:
					iv1[cTileRootName].append(0)
				idx[0][cTileRootName][0] += (iyTile,)
				idx[0][cTileRootName][1] += (ixTile,)					
			if (len(nClasses) > 1) & (cChannel == 1) :
				if nClasses[1] > 0:
					if nThresh != '':
						newProb = class_all[nClasses[cChannel]-1] / nThresh[cChannel] * 0.5
						if newProb > 0.5:
							newProb = (newProb - 0.5) / (1.0 / nThresh[cChannel] - 1) + 0.5 
						iv2[cTileRootName].append(newProb)
						if class_all[nClasses[cChannel]-1] > nThresh[cChannel]:
							count_tiles[cTileRootName][cChannel] = count_tiles[cTileRootName][cChannel] + 1
					else:
						iv2[cTileRootName].append(class_all[nClasses[1]-1])
						if class_all[nClasses[cChannel]-1] == max(class_all):
							count_tiles[cTileRootName][cChannel] = count_tiles[cTileRootName][cChannel] + 1

				else:
					iv2[cTileRootName].append(0)
				idx[1][cTileRootName][0] += (iyTile,)
				idx[1][cTileRootName][1] += (ixTile,)
			if (len(nClasses) > 2) & (cChannel == 2) :
				if nClasses[2] > 0:
					if nThresh != '':
						newProb = class_all[nClasses[2]-1] / nThresh[cChannel] * 0.5
						if newProb > 0.5:
							newProb = (newProb - 0.5) / (1.0 / nThresh[cChannel] - 1) + 0.5 
						iv3[cTileRootName].append(newProb)
						if class_all[nClasses[cChannel]-1] > nThresh[cChannel]:
							count_tiles[cTileRootName][cChannel] = count_tiles[cTileRootName][cChannel] + 1
					else:
						iv3[cTileRootName].append(class_all[nClasses[2]-1])
						if class_all[nClasses[cChannel]-1] == max(class_all):
							count_tiles[cTileRootName][cChannel] = count_tiles[cTileRootName][cChannel] + 1

				else:
					iv3[cTileRootName].append(0)
				idx[2][cTileRootName][0] += (iyTile,)
				idx[2][cTileRootName][1] += (ixTile,)
			
	file1 = open(os.path.join(FLAGS.output_dir,"distribution.txt"),"a")
	file1.write("image\tclass " + str(nClasses[0]) + "\tclass " + str(nClasses[1]) + "\tclass " + str(nClasses[2]) + "\n") 
	for slide in ListSlideNames.keys(): 
		print("slide: " + slide)
		if len(idx[0][slide][0]) > 0:
			M1y = max(idx[0][slide][0])+1
			M1x = max(idx[0][slide][1])+1
		else:
			M1y = 0
			M1x = 0
		if len(idx[1][slide][0]) > 0:
			M2y = max(idx[1][slide][0])+1
			M2x = max(idx[1][slide][1])+1
		else:
			M2y = 0
			M2x = 0
		if len(idx[2][slide][0]) > 0:
			M3y = max(idx[2][slide][0])+1
			M3x = max(idx[2][slide][1])+1
		else:
			M3y = 0
			M3x = 0

		req_yLength = max(FLAGS.tiles_size, M1y, M2y, M3y )
		req_xLength = max(FLAGS.tiles_size, M1x, M2x, M3x )
		for cChannel in range(3):
			if cChannel ==0 :
				X1 = np.zeros([req_yLength,req_xLength])

				X1[idx[cChannel][slide]] = iv1[slide]
			elif cChannel == 1 :
				X2 = np.zeros([req_yLength,req_xLength])
				if len(nClasses) >1 :
					X2[idx[cChannel][slide]] = iv2[slide]
			elif cChannel == 2 :
				X3 = np.zeros([req_yLength,req_xLength])
				if len(nClasses) >2:
					X3[idx[cChannel][slide]] = iv3[slide]
		rgb = np.zeros((req_yLength, req_xLength, 3), dtype=np.uint8)
		rgb[..., 0] = X1 * 255
		rgb[..., 1] = X2 * 255
		rgb[..., 2] = X3 * 255
		filename = os.path.join(FLAGS.output_dir,"CMap_" + slide + ".jpg")
		imsave(filename, rgb)
		file1.write(slide + "\t" + str(count_tiles[slide][0] ) + "\t" + str(count_tiles[slide][1] ) + "\t" + str(count_tiles[slide][2] ) + "\n")
	file1.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  
  parser.add_argument(
      '--tiles_size',
      type=int,
      default=-1,
      help='tile size in pixels (resulting image padded to this size if smaller).'
  )
  
  parser.add_argument(
      '--output_dir',
      type=str,
      default='mustbedefined',
      help='Output directory.'
  )
  parser.add_argument(
      '--tiles_stats',
      type=str,
      default='',
      help='text file where tile statistics are saved; if different files for different channel, sperate them with comma'
  )
  parser.add_argument(
      '--slide_filter',
      type=str,
      default='',
      help='process only images with this basename.'
  )
  
  parser.add_argument(
      '--Classes',
      type=str,
      default=None,
      help='Which classes to use  for each channel (up to 3 numbers, one per channel; first class is 1, not 0 - use 0 if a channel should not be used) - string, for example: 2,1,4'
  )
  parser.add_argument(
      '--threshold',
      type=str,
      default='',
      help='apply color normalization such as threshold > 0.5 - no normalization if left empty.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  main()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)