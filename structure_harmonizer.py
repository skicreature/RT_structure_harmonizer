import fnmatch
import json
import multiprocessing
import os
import re  # noqa may use in the future
import pathlib
from functools import partial
from multiprocessing import Pool
from typing import Any  # noqa

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from openpyxl import load_workbook  # noqa
from progress.bar import Bar
from pydicom import dcmread


class TG_263_name:
    def __init__(self, name, synonyms, categories):
        self.name = name
        self.synonyms = synonyms
        self.categories = categories

    def to_dict(self):
        return {
            "name": self.name,
            "synonyms": self.synonyms,
            "categories": self.categories,
        }


class TG_263_data_list:
    def __init__(self):
        self.data_list = []

    def add(self, tg_263_name):
        # Add a TG_263_name object to the list
        self.data_list.append(tg_263_name)

    def find_by_name(self, name):
        # Find a TG_263_name object by name
        for obj in self.data_list:
            if obj.name == name:
                return obj
        return None

    def to_dict_list(self):
        # Convert the list of objects to a list of dictionaries
        return [obj.to_dict() for obj in self.data_list]


class StructureHarmonizer:
    """
    A class for harmonizing structures in medical imaging files.

    Ideally this program should be able to be run over small to medium datasets

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
    - TG_263_file_path (str): The path to the TG263 file.
    - TG_263_standard_lists (list): A list of column names in the TG263 file.
    - TG_263_ID_column (str): The column name in the TG263 file for TG263 IDs.

    - harmonized_df (pd.DataFrame): The DataFrame to store the harmonized results.

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

    ############################################################################
    def __init__(
        self,
        directory: str | pathlib.Path | os.PathLike,
        output_filename: str,
        TG_263_file_path: str | pathlib.Path | os.PathLike,
        threshold: int = 80,
        roi_json_path: str = "json_roi_dict.json",  # TODO modify this to be unique to each run if not specified by user
    ):
        self.directory = directory
        self.RTSTRUCT_pattern = "RQ*_RTSTRUCT*.DCM"  # set the regex pattern to match
        self.harmonized_csv_path = output_filename + ".csv"
        self.threshold: int = threshold
        self.roi_dict = {}
        self.closest_matches = {}
        self.roi_json_path = roi_json_path
        self.TG_263_names_to_match = {}
        self.TG_263_file_path = TG_263_file_path
        self.TG_263_df = pd.read_excel(self.TG_263_file_path)
        self.TG_263_cols_to_match = ["TG263-Primary Name", "TG-263-Reverse Order Name"]
        self.TG_263_prime_name_col = "TG263-Primary Name"
        # self.match_roi = self.match_roi_no_json
        # set the matching method to match_roi_json by default
        # if output_file already exists, load it as a harmonized_df
        if os.path.exists(self.harmonized_csv_path):
            self.harmonized_df = pd.DataFrame()
            self.load_harmonized_df()
        else:
            self.harmonized_df = pd.DataFrame(
                index=["dummy_index"], columns=["dummy_column"]
            )

    ############################################################################
    # ------------------------- Program order / sequences ----------------------
    def run_default(self):
        """
        A default program flow to be used if the user doesn't have pre-existing
        results and does not want to customize the process harmonizes structures
        querys whether user wants to review matches
        Final step is to write results to csv file
        """
        if not os.path.exists(self.roi_json_path):
            self.get_all_ROInames(
                self.roi_json_path
            )  # gets ROInames from dicom structures
        else:
            self.load_json_roi_dict(self.roi_json_path)

        self.get_all_TG263names()  # gets TG263 names from excel file
        self.update_harmonized_df()  # update harmonized df to include new TG263 and ROI names
        self.load_json_roi_dict(self.roi_json_path)
        self.run_harmonization()  # performs the matching process

        skip_review = input("Do you want to begin review? (yes/no):")
        if skip_review.lower() not in ["no", "n"]:
            self.review_matches()
        self.write_harmonization_dict_to_csv(
            output_path=self.harmonized_csv_path + r"_matches.csv"
        )

    def run_harmonization(self):  # TODO finish this and test it.
        self.harmonize_df_get_scores()
        self.write_harmonization_dict_to_csv(
            output_path=self.harmonized_csv_path + r"_scores.csv"
        )  # noqa save the scores to a csv file
        # collapse the scores to binary matches based on the threshold
        self.harmonized_df_matches = self.harmonized_df.applymap(
            lambda x: 1 if x >= self.threshold else 0
        )  # noqa
        self.write_harmonization_dict_to_csv(
            output_path=self.harmonized_csv_path + r"_matches.csv"
        )  # noqa save the matches to a csv file

    def run_get_roi_to_json(self, output_path):
        """
        This method is used to create a json file of roi names and their corresponding file paths. This is useful for debugging and for creating
        a dictionary of roi names and their corresponding file paths to be used
        in the future.
        """
        if not os.path.exists(output_path):
            with open(output_path, "w") as file:
                json.dump({}, file)
        elif not output_path.endswith(".json"):
            raise ValueError("output_path must be a JSON file")
        else:
            print("json exists proceeding to update it")
        self.get_all_ROInames(roi_json_path=output_path)
        # run get all roi, but pass output path to save the dict to a json file

    ############################################################################
    # ---------------------- Helper / Utlity Methods ---------------------------
    def file_generator(self, directory):
        """
        Generator function to yield files matching defined pattern one by one
        """
        for root, _, files in os.walk(directory):
            for file in files:
                if fnmatch.fnmatch(file, self.RTSTRUCT_pattern):
                    yield root, _, [file]

    def load_harmonized_df(self):
        """
        Loads the stored harmonization dictionary from a previous output CSV
        file.
        """
        self.harmonized_df = pd.read_csv(self.harmonized_csv_path, index_col=0)
        # Identify user-defined names
        self.user_defined_standard_names = [
            col for col in self.harmonized_df.columns if col.startswith("USER_")
        ]
        # Identify standard names
        self.standard_names = [
            col for col in self.harmonized_df.columns if not col.startswith("USER_")
        ]

    def update_harmonized_df(self):
        """
        Add columns for each key in TG263name_dict and roi_dict to harmonized_df
        if each structure exists and is not empty
        """
        if hasattr(self, "TG_263_names_to_match") and self.TG_263_names_to_match:
            # Create a dictionary with None values for new columns
            new_columns = {
                key: np.nan
                for key in self.TG_263_names_to_match.keys()
                if key not in self.harmonized_df.columns
            }
            # Add new columns to the DataFrame
            self.harmonized_df = self.harmonized_df.assign(**new_columns)

        # Update harmonized_df with keys from roi_dict if it exists and is not empty
        if hasattr(self, "roi_dict") and self.roi_dict:
            # Create a list of current indices and new indices
            new_indices = self.harmonized_df.index.tolist() + [
                key
                for key in self.roi_dict.keys()
                if key not in self.harmonized_df.index
            ]
            # Reindex the DataFrame to include new rows
            self.harmonized_df = self.harmonized_df.reindex(new_indices)

    def load_json_roi_dict(self, roi_dict_path):
        """
        Loads a json that represents the roi_dict
        """
        with open(roi_dict_path, "r") as file:
            self.roi_dict = json.load(file)

    def write_harmonization_dict_to_csv(self, output_path=None):
        """
        Save the DataFrame as a CSV file
        """
        if output_path is None:
            output_path = self.harmonized_csv_path
        self.harmonized_df.to_csv(output_path)

    ############################### Primary Methods ############################
    # ---------------------------- Get Names -----------------------------------
    def get_all_TG263names(self):
        """
        Retrieves all TG263 names from the TG_263_file DataFrame and adds them
        to the TG263_names_to_match dictionary.
        This TG263name_dict is used by update_harmonized_dict to add new
        columns to the harmonized_df.
        Below is an example of the TG263_names_to_match dictionary:
        TG263_names_to_match = {TG263prime_name : [name1, name2, ....]}

        """
        # Iterate over the TG_263_file DataFrame
        for index in self.TG_263_df.index:
            # create dictionary of TG263 primary names for keys and a list of all TG263 names to match for values
            self.TG_263_names_to_match[
                self.TG_263_df.loc[index, self.TG_263_prime_name_col]
            ] = list(
                set(
                    [
                        self.TG_263_df.loc[index, col]
                        for col in self.TG_263_cols_to_match
                    ]
                )
            )

    def get_ROIname_from_file(self, file_path):
        """
        Uses dcmread to get all the roi_names from a DICOM file and put them in a
        dictionary for that file.
        In practice this is called by get_all_roi and then each file is
        processed in parallel
        """
        ds = dcmread(file_path)
        roi_names = [str(roi.ROIName) for roi in ds.StructureSetROISequence]
        roi_dict = {}  # Create a new dictionary, local variable
        for roi_name in roi_names:
            if roi_name in roi_dict:
                roi_dict[roi_name].append(file_path)
            else:
                roi_dict[roi_name] = [file_path]
        return roi_dict

    def get_all_ROInames(self, roi_json_path=None):
        """
        Retrieves all ROI names from the DICOM file matching the self.pattern and
        adds them to the roi_dict.
        On a large dataset this can take some time, so it is recommended to use
        the roi_json_path to save the roi_dict to a json file.
        """
        if not os.path.exists(self.directory):
            print("Directory does not exist")
        else:
            b = Bar("Processing", max=len(os.listdir(self.directory)))
            print(f"Processing {len(os.listdir(self.directory))} files...")
            file_paths = []
            for root, _, [file] in self.file_generator(self.directory):
                file_paths.append(os.path.join(root, file))  # used for multiprocessing
                # self.process_file(os.path.join(root, file), roi_dict=roi_dict) #used for single processing noqa
                b.next()
            with multiprocessing.Pool() as pool:  # Create a pool of workers
                roi_dicts = pool.map(
                    self.get_ROIname_from_file, file_paths
                )  # Process the files in parallel
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
            # If output_path is provided, update the JSON file
            if roi_json_path:
                with open(roi_json_path, "w") as file:
                    json.dump(self.roi_dict, file, indent=4, sort_keys=True)
            b.finish()

    # ---------------------------- New Matching Methods -------------------------
    def calculate_match_score(self, row):
        """
        This method is used to calculate the match score for each roi name using
        the fuzz.ratio method from the fuzzywuzzy package.
        It is possible that a faster better matching score could be obtained. If
        so, this method should be updated.
        """
        roi_lower = row.name.lower().strip()
        max_scores = {}
        for key, val in self.TG_263_names_to_match.items():
            max_score = 0
            for TG263name in val:
                TG263name_lower = TG263name.lower().strip()
                score = fuzz.ratio(TG263name_lower, roi_lower)
                if score > max_score:
                    max_score = score
            max_scores[key] = max_score
        return pd.Series(max_scores)

    def harmonize_df_get_scores(self):
        """
        This method applies the calculate_match_score method to the harmonized_df
        and returns the DataFrame with the scores for each ROI name, resulting in
        a n x m array where n is the number of ROI names and m is the number of
        TG263 names.
        """
        self.harmonized_df = self.harmonized_df.assign(
            **self.harmonized_df.apply(self.calculate_match_score, axis=1)
        )

    def get_closest_matches(self):
        """
        using the harmonized_df, find the closest matches for each roi name
        """

    # ---------------------------- Review --------------------------------------
    def review_matches(self):
        """
        Allows the user to review and confirm the matches for each ROI name,
        and add custom matches if necessary.
        Review is ordered by score, with the highest score first.
        If a match is incorrect, the user can skip to the next match for the
        same ROI name, or skip to the next ROI name.
        If a match is correct, the user can confirm the match and move on to
        the next ROI name.
        If no match is found, the user can add a custom match and move on to
        the next ROI name.
        TODO improve the custom match names, and match the custom name to
        TG263 names if possible
        #TODO find a way to prioritize matching ROI names that are most likely
        to be targets, then second priority to OARs, and lastly to other
        structures such as optimization structures.
        #TODO integrate this with the GUI, to improve review process
        """
        # TODO intergrate this with the GUI, to improve review process
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
                if user_input.lower() in ["yes", "y"]:
                    self.harmonized_df.loc[roi, TG263_id] = 1
                    break  # Move on to the next ROI
                elif user_input.lower() in ["no", "n"]:
                    print("Match is incorrect. Trying next closest match...")
                    self.harmonized_df.loc[roi, TG263_id] = 0
                    self.closest_matches[roi].pop(
                        0
                    )  # Present the next match for the same ROI
                elif user_input.lower() in ["skip", "s", "next", "n", ""]:
                    print("Skipping review of this item.")
                    break  # Move on to the next ROI
                elif user_input.lower() in [
                    "end",
                    "e",
                    "q",
                    "q()",
                    "quit",
                    "exit",
                    "stop",
                ]:
                    print("Ending review process.")
                    return  # End the entire review process
                else:
                    # currently triggers if any non-recognized input is entered
                    custom_name = user_input
                    self.user_defined_standard_names.append(custom_name)
                    self.harmonized_df.loc[roi, "USER_" + custom_name] = 1
                    print(f"Added custom match name '{custom_name}' for {roi}.")
                    break  # Move on to the next ROI

    ############################################################################


if __name__ == "__main__":
    # set the directory to search
    # directory = '/pct_ids/users/wd982598/test_requite'
    """
    directory = '/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/prostate_test/test_requite'
    output_file ='/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/structure_harmonization/structure_dictionary.csv'
    TG_263_file_path= '/home/sacketjj/Jupyter-sandbox/REQUITE_Prostate/structure_harmonization/TG263_Nomenclature_Worksheet_20170815.xls'
    # Define the match quality threshold
    threshold = 88
    # Create an instance of StructureHarmonizer and call the harmonize_structures method
    harmonizer = StructureHarmonizer(directory, output_file, TG_263_file_path, threshold=threshold)
    harmonizer.run_harmonize_dict()
    """
