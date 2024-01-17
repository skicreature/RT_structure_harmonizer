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
import json
from progress.bar import Bar

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i+n]

class StructureHarmonizer:
    """
    A class for harmonizing structures in medical imaging files.

    Parameters:
    - directory (str): The directory containing the medical imaging files.
    - output_file (str): The path to the output CSV file.
    - TG_263_file_path (str): The path to the TG263 file.
    - threshold (int, optional): The threshold for matching ROI names to TG263 names. Defaults to 80.

    Attributes:
    - directory (str): The directory containing the medical imaging files.
    - pattern (str): The regex pattern to match the medical imaging files.
    - harmonized_csv_path (str): The path to the output CSV file.
    - threshold (int): The threshold for matching ROI names to TG263 names.
    - roi_dict (dict): A dictionary to store ROI names and their corresponding file paths.
    - closest_matches (dict): A dictionary to store the closest matches for each ROI name.
    - harmonized_df (pd.DataFrame): The DataFrame to store the harmonized results.
    - TG_263_file_path (str): The path to the TG263 file.
    - TG_263_standard_lists (list): A list of column names in the TG263 file.
    - TG_263_ID_column (str): The column name in the TG263 file for TG263 IDs.

    Methods:
    - run_harmonize_dict(): Runs the harmonization process, review process, and writes the results to CSV.
    - harmonize_structures(): Performs the automated portion of the harmonization process.
    - file_generator(directory): A generator function to yield medical imaging files one by one.
    - load_stored_dictionary(): Loads the stored harmonization dictionary from the output CSV file.
    - update_stored_dictionary(): Updates the stored harmonization dictionary with new columns and rows.
    - process_file(file_path, roi_dict): Processes a medical imaging file and updates the ROI dictionary.
    - write_harmonization_dict_to_csv(): Writes the harmonization dictionary to the output CSV file.
    - get_all_TG263names(TG_263_file): Extracts all TG263 names from the TG263 file.
    - get_all_roi(): Extracts all ROI names from the medical imaging files.
    - get_roi_paths(output_path): Extracts ROI names and their corresponding file paths and saves them to a JSON file.
    - match_roi(roi, TG_263_file): Matches an ROI name to TG263 names and updates the harmonization dictionary.
    - get_closest_matches(TG_263_file): Finds the closest matches for each ROI name.
    - review_matches(): Allows the user to review and confirm the matches.
    """
    def __init__(self, directory: str, output_file: str, TG_263_file_path: str, threshold:int=80):
        self.directory = directory
        self.pattern = 'RQ*_RTSTRUCT*.DCM'     # set the regex pattern to match
        self.harmonized_csv_path = output_file
        self.threshold = threshold
        self.roi_dict = {}
        self.closest_matches = {}
        #if output_file already exists, load it as a harmonized_df
        if os.path.exists(self.harmonized_csv_path):
            self.load_harmonized_dict()
        else:
            self.harmonized_df = pd.DataFrame(columns=['dummy_column'])
        self.TG_263_file_path = TG_263_file_path
        self.TG_263_standard_lists = [ 'TG263-Primary Name', 'TG-263-Reverse Order Name']
        self.TG_263_standard_name_col = 'TG263-Primary Name'
        self.TG_263_FMAID_col = 'FMAID'


    # ------------- Program Flow Methods -------------
    def run_harmonize_dict(self):
        '''
        A default program flow to be used if the user doesn't want to customize the process
        harmonizes structures, querys whether user wants to review matches, and then writes results to csv file
        '''
        self.harmonize_structures()
        skip_review = input("Do you want to begin the review process? (yes/no): ")
        if skip_review.lower() not in ['no', 'n']:
            self.review_matches()
        self.write_harmonization_dict_to_csv()

    def harmonize_structures(self):
        '''
        Does not perform review process, this is the purely automated portion of the program for identifying matches. It is necessary to run this before review_matches() can be run as it creates the initial dictionary of roi names and TG263 names with their matches.
        '''
        self.get_all_roi()
        self.update_harmonized_dict()
        TG_263_file = pd.read_excel(self.TG_263_file_path)
        self.get_all_TG263names(TG_263_file)
        self.update_harmonized_dict()
        self.get_closest_matches(TG_263_file) # finds the closest matches for each roi name, and stores them in a dictionary and keeps track of the closest matches in a list with fuzzy ratio scores

    def get_roi_paths(self, output_path):
        if not os.path.exists(output_path):
            with open(output_path, "w") as file:
                json.dump({}, file)
        elif not output_path.endswith('.json'):
            raise ValueError("output_path must be a JSON file")
        else:
            print('json exists proceeding to update it')
        self.get_all_roi(output_path=output_path) #run get all roi, but pass output path so it will save the dictionary to a json file

    # ------------- Helper / Utlity Methods -------------
    def file_generator(self, directory):
        '''
        Generator function to yield files matching defined pattern one by one
        '''
        for root, _, files in os.walk(directory):
            for file in files:
                if fnmatch.fnmatch(file, self.pattern):
                    yield root, _, [file]

    def load_harmonized_dict(self):
        self.harmonized_df = pd.read_csv(self.harmonized_csv_path, index_col=0)
        # Identify user-defined names
        self.user_defined_standard_names = [col for col in self.harmonized_df.columns if col.startswith('USER_')]
        # Identify standard names
        self.standard_names = [col for col in self.harmonized_df.columns if not col.startswith('USER_')]

    def update_harmonized_dict(self):
        # Add columns for each key in TG263name_dict to harmonized_df if it exists and is not empty
        if hasattr(self, 'TG263name_dict') and self.TG263name_dict:
            for key in self.TG263name_dict.keys():
                if key not in self.harmonized_df.columns:
                    self.harmonized_df.loc[:,key] = None
        # Update harmonized_df with keys from roi_dict if it exists and is not empty
        if hasattr(self, 'roi_dict') and self.roi_dict:
            for key in self.roi_dict.keys():
                if key not in self.harmonized_df.index:
                    self.harmonized_df.loc[key] = None

    def load_roi_dict_from_json(self, roi_dict_path):
        with open(roi_dict_path, "r") as file:
            self.roi_dict = json.load(file)

    def process_file(self, file_path):
        # Process the file and return the ROI names
        ds = dcmread(file_path)
        roi_names = [str(roi.ROIName) for roi in ds.StructureSetROISequence]
        # Create a new dictionary
        roi_dict = {}
        for roi_name in roi_names:
            if roi_name in roi_dict:
                roi_dict[roi_name].append(file_path)
            else:
                roi_dict[roi_name] = [file_path]
        return roi_dict

    def write_harmonization_dict_to_csv(self):
        # Save the DataFrame as a CSV file
        self.harmonized_df.to_csv(self.harmonized_csv_path)

    # ------------- MAIN Methods -------------
    def get_all_TG263names(self, TG_263_file):
        # Create a dictionary to store new columns
        self.TG263name_dict = {}
        # Iterate over the TG_263_file DataFrame
        for index in TG_263_file.index:
            for column in self.TG_263_standard_lists, :
                TG263name = TG_263_file.loc[index, column]  # get the name from the TG263 file
                if pd.isnull(TG263name):  # skip if the name is NaN
                    continue
                # Add the name to TG263name_dict if not already present in harmonized_df
                if TG263name not in self.harmonized_df.columns:
                    self.TG263name_dict[TG263name] = None

    def get_all_roi(self, output_path=None, every_n_files: int=10):
        if not os.path.exists(self.directory):
            print('Directory does not exist')
        else:
            b = Bar('Processing', max=len(os.listdir(self.directory)))
            print(f"Processing {len(os.listdir(self.directory))} files...")
            file_paths = []
            for root, _, [file] in self.file_generator(self.directory):
                file_paths.append(os.path.join(root, file)) #used for multiprocessing
                #self.process_file(os.path.join(root, file), roi_dict=roi_dict) #used for single processing
                b.next()

            # Create a pool of workers
            with multiprocessing.Pool() as pool:
                # Loop over the file generator
                roi_dicts = pool.map(self.process_file, file_paths)
            # Merge the dictionaries
            self.roi_dict = {}
            for roi_dict in roi_dicts:
                for roi_name, file_paths in roi_dict.items():
                    if roi_name in self.roi_dict:
                        self.roi_dict[roi_name].extend(file_paths)
                    else:
                        self.roi_dict[roi_name] = file_paths
            # Close the pool and wait for all the workers to finish
            pool.close()
            pool.join()
            # If output_path is provided, update the JSON file one last time
            if output_path:
                with open(output_path, "w") as file:
                    json.dump(self.roi_dict, file, indent=4, sort_keys=True)
            b.finish()

    def match_roi(self, roi, TG_263_file):
        '''
        input: single roi name, and the TG263 file
        TODO figure out why matches above threshold are not getting recorded, yet they are being skipped during review
        this is where the matching happens and if a match is found, it is added to the harmonized_df. If functioning correctly then entries to the dictionary should be unique with no duplicate roi names or TG263 names
        '''
        roi_lower = roi.lower().strip()
        roi_words = set(roi_lower.split())
        closest_matches = {}
        for index in TG_263_file.index:
            for column in self.TG_263_standard_lists:
                TG263name = TG_263_file.loc[index, column]  # get the name to match from the TG263 file
                if pd.isnull(TG263name):  # skip if the name is NaN
                    continue
                TG263_id = TG_263_file.loc[index, self.TG_263_standard_name_col]  # Get the TG263 ID
                # FMAID = TG_263_file.loc[index, self.TG_263_FMAID_col] #TODO, IMPORTANT integrate FMAID
                TG263name_lower = TG263name.lower().strip()
                TG263name_words = set(TG263name_lower.split())

                # check to make sure the harmonized_df exists and is not empty before checking it and that TG263_id exists in the harmonized_df
                # Skip the matching process if the item already has a match in the harmonized_df as indicated by a 1 in the column
                # if not self.harmonized_df.empty and TG263_id in self.harmonized_df:
                #if ((not self.harmonized_df.empty) and TG263_id in self.harmonized_df.columns) and (roi in self.harmonized_df.index and self.harmonized_df.loc[roi, TG263_id] == 1):
                #    continue

                score = fuzz.ratio(TG263name_lower, roi_lower)  # Calculate the match score
                if score >= self.threshold:
                    # Add the TG263 ID as a column name to the DataFrame and identify it as a match to the corresponding roi
                    self.harmonized_df.loc[roi, TG263_id] = 1
                else:
                    # Store all matches in a list sorted by score, to be used for review_matches()
                    if roi not in closest_matches: # If the list doesn't exist yet, create it
                        closest_matches[roi] = [(TG263name, score, TG263_id)]
                    else:
                        closest_matches[roi].append((TG263name, score, TG263_id)) # Append the match to the list
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
            if self.harmonized_df.loc[roi].any() == 1:
                continue
            while self.closest_matches[roi]:
                name, score, TG263_id = self.closest_matches[roi][0]
                match_key = (roi, name, score)
                # Obtain user input if the match_key is not already in user_responses
                if match_key in user_responses:
                    user_input = user_responses[match_key]
                else:
                    print(f"Closest match for {roi} is {name} with score {score}.")
                    user_input = input("Is this match correct? (yes/no/skip/end): ")
                    user_responses[match_key] = user_input
                # check the user input and update the harmonized_df accordingly if a match is found, or end the loop if the user wants to stop
                if user_input.lower() in ['yes', 'y']:
                    self.harmonized_df.loc[roi, TG263_id] = 1
                    break  # Move on to the next ROI
                elif user_input.lower() in ['no', 'n']:
                    print("Match is incorrect. Trying next closest match...")
                    self.harmonized_df.loc[roi, TG263_id] = 0
                    self.closest_matches[roi].pop(0)  # Present the next match for the same ROI
                elif user_input.lower() in ['skip', 's', 'next', 'n','']:
                    print("Skipping review of this item.")
                    break  # Move on to the next ROI
                elif user_input.lower() in ['end','e','q','q()','quit','exit','stop']:
                    print("Ending review process.")
                    return  # End the entire review process
                else:
                    custom_name = user_input
                    self.user_defined_standard_names.append(custom_name)
                    self.harmonized_df.loc[roi, 'USER_' + custom_name] = 1
                    print(f"Added custom match name '{custom_name}' for {roi}.")
                    break  # Move on to the next ROI

if __name__ == '__main__':
    # set the directory to search
    # directory = '/pct_ids/users/wd982598/test_requite'
    directory = '/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/prostate_test/test_requite'
    output_file ='/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/structure_harmonization/structure_dictionary.csv'
    TG_263_file_path= '/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/structure_harmonization/TG263_Nomenclature_Worksheet_20170815.xls'


    # Define the match quality threshold
    threshold = 88

    # Create an instance of StructureHarmonizer and call the harmonize_structures method
    harmonizer = StructureHarmonizer(directory, output_file, TG_263_file_path, threshold=threshold)
    harmonizer.run_harmonize_dict()
