import os
import fnmatch
import pydicom
from pydicom import dcmread
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from fuzzywuzzy import fuzz

class StructureHarmonizer:
    def __init__(self, directory, output_file, pattern, TG_263_file_path, threshold=80):
        self.directory = directory
        self.output_file = output_file
        self.pattern = pattern
        self.threshold = threshold
        self.roi_dict = {}
        self.closest_matches = {}
        self.stored_dictionary:pd.DataFrame = pd.DataFrame() # Initialize an empty DataFrame to store the results
        self.TG_263_file_path = TG_263_file_path
        self.TG_263_column_names = [ 'TG263-Primary Name', 'TG-263-Reverse Order Name']
        self.TG_263_ID_column = 'TG263-Primary Name'

    def run(self):
        # A default program flow to be used if the user doesn't want to customize the process
        # harmonizes structures, querys whether user wants to review matches, and then writes results to csv

        # if output_file already exists, load it as a stored_dictionary
        if os.path.exists(self.output_file):
            self.load_stored_dictionary()
        else:
            self.stored_dictionary = pd.DataFrame()

        self.harmonize_structures()
        # Ask the user if they want to skip the review process
        skip_review = input("Do you want to begin the review process? (yes/no): ")
        if skip_review.lower() not in ['no', 'n']:
            self.review_matches()
        # Save the DataFrame as a CSV file
        self.write_to_csv()

    def harmonize_structures(self):
        self.get_all_roi()
        TG_263_file = pd.read_excel(self.TG_263_file_path)
        self.get_all_TG263names(TG_263_file, self.TG_263_column_names)
        self.get_closest_matches(TG_263_file, self.TG_263_column_names)

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
        # Iterate over the TG_263_file DataFrame
        for index in TG_263_file.index:
            for column in self.TG_263_column_names:
                TG263name = TG_263_file.loc[index, column]  # get the name from the TG263 file
                if pd.isnull(TG263name):  # skip if the name is NaN
                    continue
                # Add the name to the stored_dictionary if not already present
                if TG263name not in self.stored_dictionary.columns:
                    self.stored_dictionary.loc[:, TG263name] = 0

    def get_all_roi(self):
        # this function will search the directory for RTSTRUCT files and create a new dictionary of ROI names and file names, this can later be compared to the stored_dictionary
        if not os.path.exists(self.directory):
            print('Directory does not exist')
        else:
            # perform the recursive search
            for root, _, files in self.file_generator(self.directory, self.pattern):
                for file in files:
                    # read the file
                    ds = dcmread(os.path.join(root, file))
                    for roi in ds.StructureSetROISequence:
                        # get the ROIName as a string
                        roi_name = str(roi.ROIName)
                        # check if the ROIName is already in the dictionary
                        if roi_name in self.roi_dict:
                            # if it is, append the file name to the value list
                            self.roi_dict[roi_name].append(file)
                        else:
                            # if not, add the ROIName to the dictionary with the file name as the value
                            self.roi_dict[roi_name] = [file]
                        if roi_name not in self.stored_dictionary.index:
                            self.stored_dictionary.loc[roi_name, :] = 0

    def get_closest_matches(self, TG_263_file):
        # Initialize an empty dictionary to keep track of the closest matches
        self.closest_matches = {}

        # Fill the DataFrame
        for index in TG_263_file.index:
            for column in self.TG_263_column_names:
                TG263name = TG_263_file.loc[index, column]  # get the name to match from the TG263 file
                if pd.isnull(TG263name):  # skip if the name is NaN
                    continue

                TG263_id = TG_263_file.loc[index, self.TG_263_ID_column]  # Get the TG263 ID
                TG263name_lower = TG263name.lower().strip()  # Convert to lowercase and strip spaces
                TG263name_words = set(TG263name_lower.split()) # Split the name into set of words
                for roi in self.roi_dict:
                    roi_lower = roi.lower().strip()  # Convert to lowercase and strip spaces
                    roi_words = set(roi_lower.split())

                    # check to make sure the stored_dictionary exists and is not empty before checking it and that TG263_id exists in the stored_dictionary
                    if not self.stored_dictionary.empty and TG263_id in self.stored_dictionary:
                        # Skip the matching process if the item already has a match in the output file
                        if roi in self.stored_dictionary.index and self.stored_dictionary.loc[roi, TG263_id] == 1:
                            continue

                    score = fuzz.ratio(TG263name_lower, roi_lower)  # Calculate the match score
                    # If the score is above the threshold, add the match to the DataFrame
                    if score >= self.threshold:
                        # Add the TG263 ID as a column name to the DataFrame and identify it as a match to the corresponding roi
                        self.stored_dictionary.loc[roi, TG263_id] = 1
                        # Once a match is found, stop checking other ROIs for this TG263name
                        break
                    else:
                        # Store all matches in a list sorted by score, to be used for review_matches()
                        if roi not in self.closest_matches: # If the list doesn't exist yet, create it
                            self.closest_matches[roi] = [(TG263name, score)]
                        else:
                            self.closest_matches[roi].append((TG263name, score)) # Append the match to the list
                            self.closest_matches[roi].sort(key=lambda x: x[1], reverse=True) # Sort by score

    def review_matches(self):
        # Initialize a dictionary to store user responses
        user_responses = {}
        # Initialize a list to store user defined standard names
        self.user_defined_standard_names = []

        # Review the closest matches
        for roi in self.closest_matches.keys():
            # If a match has already been recorded for this ROI, skip it
            if self.stored_dictionary.loc[roi].any() == 1:
                continue
            for column in self.stored_dictionary.columns:
                while self.closest_matches[roi]:
                    name, score = self.closest_matches[roi][0]
                    # Create a key to identify the match
                    match_key = (roi, name, score)
                    if match_key in user_responses:
                        # If the match has already been reviewed, use the stored response
                        user_input = user_responses[match_key]
                    else:
                        print(f"Closest match for {roi} is {name} with score {score}.")
                        user_input = input("Is this match correct? (yes/no/skip/end): ")
                        # Store the user's response
                        user_responses[match_key] = user_input

                    if user_input.lower() in ['yes', 'y']:
                        self.stored_dictionary.loc[roi, column] = 1
                        break
                    elif user_input.lower() in ['no', 'n']:
                        print("Match is incorrect. Trying next closest match...")
                        self.stored_dictionary.loc[roi, column] = 0
                        self.closest_matches[roi].pop(0)
                    elif user_input.lower() in ['skip', 's']:
                        print("Skipping review of this item.")
                        break
                    elif user_input.lower() == 'end':
                        print("Ending review process.")
                        break
                    else:
                        # Treat the user input as a custom match name
                        custom_name = user_input
                        self.user_defined_standard_names.append(custom_name)
                        # Add the custom name to the DataFrame with a prefix
                        self.stored_dictionary.loc[roi, 'USER_' + custom_name] = 1
                        print(f"Added custom match name '{custom_name}' for {roi}.")
                        break
                if user_input.lower() == 'end':
                    break

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
    harmonizer.run()
