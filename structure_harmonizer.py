import os
import fnmatch
import pydicom
from pydicom import dcmread
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from fuzzywuzzy import fuzz
import re
import multiprocessing
from multiprocessing import Pool, Manager
from functools import partial

class StructureHarmonizer:
    def __init__(self, directory, output_file, pattern, TG_263_file_path, threshold=80):
        self.directory = directory
        self.output_file = output_file
        self.pattern = pattern
        self.threshold = threshold
        self.roi_dict = {}
        self.closest_matches = {}
        #if output_file already exists, load it as a stored_dictionary
        if os.path.exists(self.output_file):
            self.load_stored_dictionary()
        else:
            self.stored_dictionary = pd.DataFrame(columns=['dummy_column'])
        self.TG_263_file_path = TG_263_file_path
        self.TG_263_column_names = [ 'TG263-Primary Name', 'TG-263-Reverse Order Name']
        self.TG_263_ID_column = 'TG263-Primary Name'

    def run_dictionary(self):
        # A default program flow to be used if the user doesn't want to customize the process
        # harmonizes structures, querys whether user wants to review matches, and then writes results to csv
        self.harmonize_structures()
        # Ask the user if they want to skip the review process
        skip_review = input("Do you want to begin the review process? (yes/no): ")
        if skip_review.lower() not in ['no', 'n']:
            self.review_matches()
        # Save the DataFrame as a CSV file
        self.write_to_csv()

    def harmonize_structures(self):
        # does not perform review process, this is the purely automated portion of the program for identifying matches. It is necessary to run this before review_matches() can be run as it creates the initial dictionary of roi names and TG263 names with their matches.
        self.get_all_roi()
        TG_263_file = pd.read_excel(self.TG_263_file_path)
        self.get_all_TG263names(TG_263_file)
        self.get_closest_matches(TG_263_file) # finds the closest matches for each roi name, and stores them in a dictionary and keeps track of the closest matches in a list with fuzzy ratio scores

    def file_generator(self, directory, pattern):
        # Generator function to yield files one by one
        for root, _, files in os.walk(directory):
            for file in files:
                if fnmatch.fnmatch(file, pattern):
                    yield root, _, [file]

    def load_stored_dictionary(self):
        self.stored_dictionary = pd.read_csv(self.output_file, index_col=0)
        # Identify user-defined names
        self.user_defined_standard_names = [col for col in self.stored_dictionary.columns if col.startswith('USER_')]
        # Identify standard names
        self.standard_names = [col for col in self.stored_dictionary.columns if not col.startswith('USER_')]

    def get_all_TG263names(self, TG_263_file):
        # Create a dictionary to store new columns
        self.TG263name_dict = {}

        # Iterate over the TG_263_file DataFrame
        for index in TG_263_file.index:
            for column in self.TG_263_column_names:
                TG263name = TG_263_file.loc[index, column]  # get the name from the TG263 file
                if pd.isnull(TG263name):  # skip if the name is NaN
                    continue
                # Add the name to TG263name_dict if not already present in stored_dictionary
                if TG263name not in self.stored_dictionary.columns:
                    self.TG263name_dict[TG263name] = None

        self.update_stored_dictionary()

    def update_stored_dictionary(self):
        # Add columns for each key in TG263name_dict to stored_dictionary if it exists and is not empty
        if hasattr(self, 'TG263name_dict') and self.TG263name_dict:
            for key in self.TG263name_dict.keys():
                if key not in self.stored_dictionary.columns:
                    self.stored_dictionary.loc[:,key] = None

        # Update stored_dictionary with keys from roi_dict if it exists and is not empty
        if hasattr(self, 'roi_dict') and self.roi_dict:
            for key in self.roi_dict.keys():
                if key not in self.stored_dictionary.index:
                    self.stored_dictionary.loc[key] = None

    def process_file(self, file_path, roi_dict):
        # Process the file and return the ROI names
        ds = dcmread(file_path)
        roi_names = [str(roi.ROIName) for roi in ds.StructureSetROISequence]
        # Update roi_dict
        for roi_name in roi_names:
            if roi_name in roi_dict:
                roi_dict[roi_name].append(file_path)
            else:
                roi_dict[roi_name] = [file_path]
        return roi_names

    def get_all_roi(self):
        if not os.path.exists(self.directory):
            print('Directory does not exist')
        else:
            file_paths = []
            for root, _, files in self.file_generator(self.directory, self.pattern):
                for file in files:
                    file_paths.append(os.path.join(root, file))

            # Create a Manager object and a shared dictionary
            with Manager() as manager:
                roi_dict = manager.dict()
                # Create a pool of workers
                with multiprocessing.Pool() as pool:
                    # Use a partial function to pass the shared dictionary to process_file
                    process_file_partial = partial(self.process_file, roi_dict=roi_dict)

                    # Apply process_file to each file path in parallel
                    pool.map(process_file_partial, file_paths)

                # Convert the shared dictionary back to a regular dictionary
                self.roi_dict = dict(roi_dict)
        self.update_stored_dictionary()

    def match_roi(self, roi, TG_263_file):
        #this is where the matching happens and if a match is found, it is added to the stored_dictionary. If functioning correctly then entries to the dictionary should be unique with no duplicate roi names or TG263 names
        roi_lower = roi.lower().strip()
        roi_words = set(roi_lower.split())
        closest_matches = {}
        for index in TG_263_file.index:
            for column in self.TG_263_column_names:
                TG263name = TG_263_file.loc[index, column]  # get the name to match from the TG263 file
                if pd.isnull(TG263name):  # skip if the name is NaN
                    continue
                TG263_id = TG_263_file.loc[index, self.TG_263_ID_column]  # Get the TG263 ID
                TG263name_lower = TG263name.lower().strip()
                TG263name_words = set(TG263name_lower.split())

                # check to make sure the stored_dictionary exists and is not empty before checking it and that TG263_id exists in the stored_dictionary
                # Skip the matching process if the item already has a match in the stored_dictionary as indicated by a 1 in the column
                # if not self.stored_dictionary.empty and TG263_id in self.stored_dictionary:
                if ((not self.stored_dictionary.empty) and TG263_id in self.stored_dictionary.columns) and (roi in self.stored_dictionary.index and self.stored_dictionary.loc[roi, TG263_id] == 1):
                    continue

                score = fuzz.ratio(TG263name_lower, roi_lower)  # Calculate the match score
                if score >= self.threshold:
                    # Add the TG263 ID as a column name to the DataFrame and identify it as a match to the corresponding roi
                    if roi not in self.stored_dictionary.index:
                        self.stored_dictionary.loc[roi, TG263_id] = 1
                    # Once a match is found, stop checking other ROIs for this TG263name
                    break
                else:
                    # Store all matches in a list sorted by score, to be used for review_matches()
                    if roi not in closest_matches: # If the list doesn't exist yet, create it
                        closest_matches[roi] = [(TG263name, score)]
                    else:
                        closest_matches[roi].append((TG263name, score)) # Append the match to the list
                        closest_matches[roi].sort(key=lambda x: x[1], reverse=True) # Sort by score

        return closest_matches

    def get_closest_matches(self, TG_263_file):
        # Initialize an empty dictionary to keep track of the closest matches
        self.closest_matches = {}

        # Create a pool of workers
        with Pool() as pool:
            # Use a partial function to pass the TG_263_file to match_roi
            match_roi_partial = partial(self.match_roi, TG_263_file=TG_263_file)

            # Apply match_roi to each ROI in parallel
            results = pool.map(match_roi_partial, self.roi_dict.keys())

        # Combine the results
        for result in results:
            self.closest_matches.update(result)

    def review_matches(self):
        user_responses = {}
        self.user_defined_standard_names = []

        for roi in self.closest_matches.keys():
            if self.stored_dictionary.loc[roi].any() == 1:
                continue
            while self.closest_matches[roi]:
                name, score = self.closest_matches[roi][0]
                match_key = (roi, name, score)
                # Obtain user input if the match_key is not already in user_responses
                if match_key in user_responses:
                    user_input = user_responses[match_key]
                else:
                    print(f"Closest match for {roi} is {name} with score {score}.")
                    user_input = input("Is this match correct? (yes/no/skip/end): ")
                    user_responses[match_key] = user_input
                # check the user input and update the stored_dictionary accordingly if a match is found, or end the loop if the user wants to stop
                if user_input.lower() in ['yes', 'y']:
                    self.stored_dictionary.loc[roi, name] = 1
                    break  # Move on to the next ROI
                elif user_input.lower() in ['no', 'n']:
                    print("Match is incorrect. Trying next closest match...")
                    self.stored_dictionary.loc[roi, name] = 0
                    self.closest_matches[roi].pop(0)  # Present the next match for the same ROI
                elif user_input.lower() in ['skip', 's', 'next', 'n']:
                    print("Skipping review of this item.")
                    break  # Move on to the next ROI
                elif user_input.lower() in ['end','e','q','quit','exit','stop','/r/n','/r','/n']:
                    print("Ending review process.")
                    return  # End the entire review process
                else:
                    custom_name = user_input
                    self.user_defined_standard_names.append(custom_name)
                    self.stored_dictionary.loc[roi, 'USER_' + custom_name] = 1
                    print(f"Added custom match name '{custom_name}' for {roi}.")
                    break  # Move on to the next ROI

    def write_to_csv(self):
        # Save the DataFrame as a CSV file
        self.stored_dictionary.to_csv(self.output_file)

if __name__ == '__main__':
    # set the directory to search
    # directory = '/pct_ids/users/wd982598/test_requite'
    directory = '/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/prostate_test/test_requite'
    output_file ='/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/structure_harmonization/structure_dictionary.csv'
    TG_263_file_path= '/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/structure_harmonization/TG263_Nomenclature_Worksheet_20170815.xls'
    # set the regex pattern to match
    pattern = '*RTSTRUCT*.DCM'


    # Define the match quality threshold
    threshold = 88

    # Create an instance of StructureHarmonizer and call the harmonize_structures method
    harmonizer = StructureHarmonizer(directory, output_file, pattern, TG_263_file_path, threshold=threshold)
    harmonizer.run_dictionary()
