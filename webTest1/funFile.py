# This file is fun because it is very fun
import pathlib
import os

def get_list_of_files(folder):
	list_of_files = []
	for file in os.listdir(folder):
		list_of_files.append(file)
	
	return list_of_files

