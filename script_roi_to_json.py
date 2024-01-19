import os
import fnmatch
import pydicom
from pydicom import dcmread
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from fuzzywuzzy import fuzz
import re
from structure_harmonizer import StructureHarmonizer
#from structure_regex import generate_structure_name

# set the directory to search
# directory = '/pct_ids/users/wd982598/test_requite'
directory = '/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/prostate_test/test_requite'
output_file ='/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/structure_harmonization/structure_dictionary.csv'
TG_263_file_path= '/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/structure_harmonization/TG263_Nomenclature_Worksheet_20170815.xls'
#directory = '/mnt/t/Batch_1/'

# Define the match quality threshold
threshold = 88

# Create an instance of StructureHarmonizer and call the harmonize_structures method
harmonizer = StructureHarmonizer(directory, output_file, TG_263_file_path, threshold=threshold)

harmonizer.get_roi_paths('/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/structure_harmonization/structure_json_file.json')