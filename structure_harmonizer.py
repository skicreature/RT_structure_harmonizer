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
        self.stored_dictionary = pd.DataFrame()
        self.TG_263_file_path = TG_263_file_path
        self.TG_263_column_names = ['TG-263-Reverse Order Name', 'TG263-Primary Name']

    def get_roi_dict(self):
        # initialize the dictionary
        self.roi_dict = {}

        if not os.path.exists(self.directory):
            print('Directory does not exist')
        else:
            # perform the recursive search
            for root, _, files in os.walk(self.directory):
                for file in files:
                    if fnmatch.fnmatch(file, self.pattern):
                        # read the file
                        ds = dcmread(os.path.join(root, file))

                        for roi in ds.StructureSetROISequence:
                            # get the ROIName
                            roi_name = roi.ROIName

                            # check if the ROIName is already in the dictionary
                            if roi_name in self.roi_dict:
                                # if it is, append the file name to the value list
                                self.roi_dict[roi_name].append(file)
                            else:
                                # if not, add the ROIName to the dictionary with the file name as the value
                                self.roi_dict[roi_name] = [file]

    def get_closest_matches(self, TG_263_file, columns_with_names):
        # Initialize a dictionary to keep track of the closest matches
        self.closest_matches = {}

        # Fill the DataFrame
        for index in TG_263_file.index:
            for column in columns_with_names:
                name = TG_263_file.loc[index, column]  # Keep the original name
                name_lower = name.lower().strip()  # Convert to lowercase and strip spaces
                name_words = set(name_lower.split())
                for roi in self.roi_dict:
                    roi_lower = roi.lower().strip()  # Convert to lowercase and strip spaces
                    roi_words = set(roi_lower.split())
                    # Skip the matching process if the item already has a match in the output file
                    if roi in self.stored_dictionary.index and self.stored_dictionary.loc[roi, column] == 1:
                        continue
                    # Check for subset match first
                    if name_words.issubset(roi_words) or roi_words.issubset(name_words):
                        # If there's a subset match, use fuzzy matching to check the similarity
                        score = fuzz.ratio(name_lower, roi_lower)
                        if score >= self.threshold:
                            self.stored_dictionary.loc[roi, column] = 1
                        else:
                            # Store all matches in a list sorted by score
                            if roi not in self.closest_matches:
                                self.closest_matches[roi] = [(name, score)]
                            else:
                                self.closest_matches[roi].append((name, score))
                                self.closest_matches[roi].sort(key=lambda x: x[1], reverse=True)

    def review_matches(self, columns):
        # Ask the user if they want to skip the review process
        skip_review = input("Do you want to skip the review process? (yes/no): ")

        if skip_review.lower() not in ['yes', 'y']:
            # Review the closest matches
            for roi in self.closest_matches.keys():
                for column in columns:
                    while self.closest_matches[roi]:
                        name, score = self.closest_matches[roi][0]
                        print(f"Closest match for {roi} is {name} with score {score}.")
                        user_input = input("Is this match correct? (yes/no/skip/end): ")
                        if user_input.lower() in ['yes', 'y']:
                            self.stored_dictionary.loc[roi, column] = 1
                            break
                        elif user_input.lower() in ['no', 'n']:
                            print("Match is incorrect. Trying next closest match...")
                            self.closest_matches[roi].pop(0)
                        elif user_input.lower() in ['skip', 's']:
                            print("Skipping review of this item.")
                            break
                        elif user_input.lower() == 'end':
                            print("Ending review process.")
                            break
                        else:
                            print("Invalid input. Skipping review of this item.")
                            break
                    if user_input.lower() == 'end':
                        break
                if user_input.lower() == 'end':
                    break

        self.write_to_csv()

    def write_to_csv(self):
        # Save the DataFrame as a CSV file
        self.stored_dictionary.to_csv(self.output_file)

    def harmonize_structures(self):
        self.get_roi_dict()
        TG_263_file = pd.read_excel(self.TG_263_file_path)
        self.stored_dictionary = pd.read_csv(self.output_file, index_col=0)
        df = pd.DataFrame(index=TG_263_file.index, columns=TG_263_file.columns)
        self.get_closest_matches(TG_263_file, self.TG_263_column_names)
        self.review_matches(self.TG_263_column_names)


if __name__ == '__main__':
    # set the directory to search
    # directory = '/pct_ids/users/wd982598/test_requite'
    directory = '/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/prostate_test/test_requite'
    output_file ='./structure_harmonization/structure_dictionary.csv'
    TG_263_file_path= '/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/structure_harmonization/TG263_Nomenclature_Worksheet_20170815.xls'
    # set the regex pattern to match
    pattern = '*RTSTRUCT*.DCM'


    # Define the match quality threshold
    threshold = 80

    # Create an instance of StructureHarmonizer and call the harmonize_structures method
    harmonizer = StructureHarmonizer(directory, output_file, pattern, TG_263_file_path, threshold=threshold)
    harmonizer.harmonize_structures()
