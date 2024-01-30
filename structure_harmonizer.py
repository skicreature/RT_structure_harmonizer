import fnmatch
import json
import multiprocessing
import os
import re  # noqa may use in the future
import pathlib
from functools import partial
from multiprocessing import Pool
from typing import Any, TypeVar, Type, Optional, List, Dict, Union, cast  # noqa

import tkinter as tk
from tkinter import ttk, simpledialog
import pandas as pd
import numpy as np
import pandas as pd
from rapidfuzz import fuzz  # consider importing process to use to speed things up
from openpyxl import load_workbook  # noqa
from progress.bar import Bar
from pydicom import dcmread

from polyfuzz.models import Embeddings
from flair.embeddings import TransformerWordEmbeddings

from polyfuzz import PolyFuzz

# consider implementing polyfuzz, an alternative to fuzzywuzzy that is faster and more accurate and can use transformer based models.


################################################################################
# -------------------------- Utility Classes -----------------------------------
class BaseRTItem:
    """Base class for radiotherapy structure names"""

    def __init__(self, name: str, categories: Optional[list[str]] = None):
        self.name = name
        self.categories = categories if categories else []

    def to_dict(self) -> Dict[str, Union[str, list[str]]]:
        return self.__dict__


T = TypeVar("T", bound=BaseRTItem)


class BaseDict(dict):
    """Base class for radiotherapy structure dictionaries"""

    def __init__(self, item_class: Type[T] = BaseRTItem, *args, **kwargs):
        self.item_class = item_class
        super().__init__(*args, **kwargs)

    def add(self, item: T):  # type: ignore
        if not isinstance(item, self.item_class):
            raise TypeError(
                f"Expected instance of {self.item_class.__name__}, got {type(item).__name__}"
            )
        if item.name not in self:
            self[item.name] = item

    def get(self, name: str) -> Optional[T]:  # type: ignore
        return cast(Optional[T], self.get(name))

    def __getitem__(self, name: str) -> Optional[T]:  # type: ignore
        return cast(Optional[T], super().__getitem__(name)) if name in self else None

    def to_dict_list(self) -> List[Dict[str, Union[str, list[str]]]]:
        return [item.to_dict() for item in self.values()]

    def load_from_json(self, path: str):
        with open(path, "r") as file:
            data = json.load(file)
            for item in data:
                self.add(self.item_class(**item))

    def write_to_json(self, path: str):
        dict_list = self.to_dict_list()
        with open(path, "w") as file:
            json.dump(dict_list, file)


class TG_263_name(BaseRTItem):
    """
    A class that represents a TG-263 name.
    Parameters:
        - name: str / The primary TG-263 name.
        - synonyms: list / list of the synonyms
        - categories: list / list of categories the TG-263 name belongs to
        - FMAID: str / the FMAID of the TG-263 name
    Methods:
        - add_synonym(synonym): Adds a synonym to the list of synonyms.

    #TODO: eclipse definitions for volume types are different than TG_263 Target Types,
    this needs to be united at some point
    """

    all_synonyms = set()

    def __init__(
        self,
        name: str,
        synonyms: list[str],
        categories: list[str],
        target_type: Optional[str] = None,
        FMAID: Optional[str] = None,
    ):
        super().__init__(name, categories)
        self.synonyms = []  # add additional parameter for synonyms
        for synonym in synonyms:
            self.add_synonym(synonym)
        self.target_type = target_type if target_type else None
        self.FMAID = FMAID if FMAID else None

    def add_synonym(self, synonym: str):
        if synonym in TG_263_name.all_synonyms:
            return  # Skip adding the synonym if it already exists
        TG_263_name.all_synonyms.add(synonym)
        self.synonyms.append(synonym)

    def set_target_type(self, target_type: str):
        self.target_type = target_type


class TG_263_data_dict(BaseDict):
    """
    A dictionary that stores TG-263 names and their corresponding names to match as TG_263_name objects.
    Keys = names
    Parameters:
        - name: str / The primary TG-263 name.
        - synonyms: list / list of the synonyms
        - categories: list / list of categories the TG-263 name belongs to
        - FMAID: str / the FMAID of the TG-263 name
    Methods:
        - add_synonym(synonym): Adds a synonym to the list of synonyms.
    """

    item_class = TG_263_name

    def __init__(self, *args, **kwargs):
        super().__init__(self.item_class, *args, **kwargs)


class RTStruct(BaseRTItem):
    """
    A class that represents an RTSTRUCT name.
    Parameters:
        - name: str / The primary RTSTRUCT name.
        - roi_match: str / The string that could be used to overwrite the RTSTRUCT name in the dicom file when importing into eclipse
        - categories: list / list of categories the RTSTRUCT name belongs to
        - file_paths: list / list of file paths the RTSTRUCT where the RTSTRUCT name has been found
    """

    def __init__(
        self,
        name: str,
        roi_match: Optional[str] = None,
        categories: Optional[list[str]] = None,
        dcm_paths: Optional[list[str]] = None,
    ):
        super().__init__(name, categories if categories else [])
        self.roi_match = roi_match
        self.dcm_paths = dcm_paths if dcm_paths else []

    def set_roi_match(self, roi_match: str):
        self.roi_match = roi_match


class RTStruct_dict(BaseDict):
    """
    A dictionary for storing RTSTRUCT names and their corresponding file paths as RTSTRUCT objects. Exists for typehinting purposes
    Parameters:
        - name: str / The primary RTSTRUCT name.
        - roi_match: str / The string that could be used to overwrite the RTSTRUCT name in the dicom file when importing into eclipse
        - categories: list / list of categories the RTSTRUCT name belongs to
        - file_paths: list / list of file paths the RTSTRUCT where the RTSTRUCT name has been found
    """

    item_class = RTStruct

    def __init__(self, *args, **kwargs):
        super().__init__(self.item_class, *args, **kwargs)


################################################################################
# ---------------------------- StructureHarmonizer ----------------------------
class StructureHarmonizer:
    """
    A class that performs structure harmonization by matching ROI names from DICOM files
    with TG-263 names from an Excel file.
    Parameters:
        - directory: str | pathlib.Path | os.PathLike
            The directory path where the DICOM files are located.
        - output_filename: str
            The name of the output file for the harmonized results.
        - TG_263_file_path: str | pathlib.Path | os.PathLike
            The file path of the Excel file containing TG-263 names.
        - threshold: int, optional (default=80)
            The threshold value used to determine matches between ROI names and TG-263 names.
        - roi_json_path: str, optional (default="json_roi_dict.json")
            The file path of the JSON file to store the ROI names and their corresponding file paths.
    Attributes:
        - directory: str | pathlib.Path | os.PathLike
            The directory path where the DICOM files are located.
        - RTSTRUCT_pattern: str
            The regex pattern to match DICOM files.
        - harmonized_csv_path: str
            The file path of the output CSV file for the harmonized results.
        - threshold: int
            The threshold value used to determine matches between ROI names and TG-263 names.
        - roi_dict: dict
            A dictionary that stores the ROI names and their corresponding file paths.
        - closest_matches: dict
            A dictionary that stores the closest matches between ROI names and TG-263 names.
        - roi_json_path: str
            The file path of the JSON file to store the ROI names and their corresponding file paths.
        - TG_263_names_to_match: dict
            A dictionary that stores the TG-263 names and their corresponding names to match.
        - TG_263_file_path: str | pathlib.Path | os.PathLike
            The file path of the Excel file containing TG-263 names.
        - TG_263_df: pandas.DataFrame
            The DataFrame containing the TG-263 names from the Excel file.
        - TG_263_cols_to_match: list
            The list of column names in the TG_263_df to match.
        - TG_263_prime_name_col: str
            The column name in the TG_263_df for the TG-263 primary name.
        - harmonized_df: pandas.DataFrame
            The DataFrame that stores the harmonized results.
    Methods:
        - run_default(): Runs the default program flow for structure harmonization.
        - run_harmonization(): Performs the matching process for structure harmonization.
        - run_get_roi_to_json(output_path): Creates a JSON file of ROI names and their corresponding file paths.
        - file_generator(directory): Generator function to yield files matching the defined pattern.
        - load_harmonized_df(): Loads the stored harmonization dictionary from a previous output CSV file.
        - update_harmonized_df(): Adds columns for each key in TG263name_dict and roi_dict to harmonized_df.
        - load_json_roi_dict(roi_dict_path): Loads a JSON file that represents the roi_dict.
        - write_harmonization_dict_to_csv(output_path): Saves the DataFrame as a CSV file.
        - get_all_TG263names(): Retrieves all TG263 names from the TG_263_file DataFrame.
        - get_ROIname_from_file(file_path): Gets all the ROI names from a DICOM file.
        - get_all_ROInames(roi_json_path): Retrieves all ROI names from the DICOM files.
        - calculate_match_score(row): Calculates the match score for each ROI name.
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
        self.harmonized_csv_path = output_filename
        self.roi_dict = {}
        self.closest_matches = {}  # utilized in review_matches
        self.roi_json_path = roi_json_path
        self.TG_263_names_to_match = {}
        self.TG_263_file_path = TG_263_file_path
        self.TG_263_df = pd.read_excel(self.TG_263_file_path)
        self.TG_263_cols_to_match = ["TG263-Primary Name", "TG-263-Reverse Order Name"]
        self.TG_263_prime_name_col = "TG263-Primary Name"
        self.TG_263_cats = ["Major Category", "Minor Category", "Anatomic Group"]
        self.TG_263_target_col = "Target Type"

        # TODO: eliminate the load of harmonized_dfs once matches are stored in TG_263_data_dict based jsons
        # However, the scores will still need to be calculated and stored in the harmonized_df

        # if output_file already exists, load it as a harmonized_df
        if os.path.exists(self.harmonized_csv_path):
            self.harmonized_df = pd.DataFrame()
            self.load_harmonized_df()
        else:
            self.harmonized_df = pd.DataFrame(
                index=["dummy_index"], columns=["dummy_column"]
            )

        # uses the threshold methold right now, but could be changed to use the closest match method
        self.threshold: int = threshold
        self.match_method: function = lambda x: 1 if x >= self.threshold else 0

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
        self.harmonized_df.to_csv(self.harmonized_csv_path + r"_matches.csv")

    def run_harmonization(self):  # TODO finish this and test it.
        self.harmonize_df_get_scores()
        self.harmonized_df.to_csv(self.harmonized_csv_path + r"_scores.csv")
        # replaces the scores with new match identifiers (likely 1 or 0) based on the match method (default threshold)
        self.harmonized_df_matches: pd.DataFrame = self.harmonized_df.applymap(self.match_method)  # type: ignore # noqa
        self.harmonized_df_matches.to_csv(self.harmonized_csv_path + r"_matches.csv")

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

    def load_harmonized_df(self, harmonized_csv_path: Optional[str] = None):
        """
        Loads the stored harmonization dictionary from a previous output CSV
        file.
        """
        if not harmonized_csv_path:
            harmonized_csv_path = self.harmonized_csv_path
        self.harmonized_df = pd.read_csv(harmonized_csv_path, index_col=0)
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
        if hasattr(self, "TG_263_names_to_match") and self.TG263_data_dict:
            # Create a dictionary with None values for new columns
            new_columns = {
                key: np.nan
                for key in self.TG263_data_dict.keys()
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
        self.roi_dict = RTStruct_dict()
        self.roi_dict.load_from_json(roi_dict_path)

    ############################### Primary Methods ############################
    # ---------------------------- Get Names -----------------------------------

    def get_all_TG263names(self):
        # This works
        """
        Retrieves all TG263 names from the TG_263_file DataFrame and adds them
        to the TG263_data_dict.
        """
        # Create an instance of TG_263_data_dict
        self.TG263_data_dict = TG_263_data_dict()

        # Iterate over the TG_263_file DataFrame
        for index in self.TG_263_df.index:
            # Create a TG_263_name object for each row in the DataFrame
            name: str = cast(str, self.TG_263_df.loc[index, self.TG_263_prime_name_col])
            synonyms: list[str] = list(
                set(
                    cast(str, self.TG_263_df.loc[index, col])
                    for col in self.TG_263_cols_to_match
                )
            )
            categories: list[str] = cast(
                list[str], [self.TG_263_df.loc[index, col] for col in self.TG_263_cats]
            )
            target_type: str = cast(
                str, self.TG_263_df.loc[index, self.TG_263_target_col]
            )
            fmaid: str = cast(str, self.TG_263_df.loc[index, "FMAID"])

            tg_263_name = TG_263_name(
                name, synonyms, categories, target_type, FMAID=fmaid
            )

            # Add the TG_263_name object to the TG_263_data_dict
            self.TG263_data_dict.add(tg_263_name)

    # ---------------------------- Get ROIs ------------------------------------
    # TODO rewrite two methods below to use the RTSTRUCT and RTSTRUCT_dict classes
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

    # ---------------------------- New get ROI Methods -------------------------
    # Utilizes the RTSTRUCT, RTSTRUCT_dict, TG_263_name, and TG_263_data_dict classes
    # TODO: test these methods, only remove old methods when certain that new methods work
    # and work fast enough
    def get_ROIname_from_file_new(self, file_path):  # TODO test this method
        """
        Uses dcmread to get all the roi_names from a DICOM file and put them in a
        dictionary for that file.
        In practice this is called by get_all_roi and then each file is
        processed in parallel
        """
        ds = dcmread(file_path)
        roi_names = [str(roi.ROIName) for roi in ds.StructureSetROISequence]
        rtstruct_dict = RTStruct_dict()  # Create a new RTSTRUCT_dict
        for roi_name in roi_names:
            if rtstruct_dict[roi_name]:
                rtstruct_dict[roi_name].file_paths.append(file_path)  # type: ignore
            else:
                rtstruct = RTStruct(roi_name, dcm_paths=[file_path])
                rtstruct_dict.add(rtstruct)
        return rtstruct_dict

    def get_all_ROInames_new(self, roi_json_path=None):  # TODO test this method
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
                b.next()
            with multiprocessing.Pool() as pool:  # Create a pool of workers
                rtstruct_dicts = pool.map(
                    self.get_ROIname_from_file, file_paths
                )  # Process the files in parallel
            # Merge the dictionaries
            self.rtstruct_dict = RTStruct_dict()
            for rtstruct_dict in rtstruct_dicts:
                for rtstruct in rtstruct_dict.values():
                    if self.rtstruct_dict[rtstruct.name]:
                        self.rtstruct_dict[rtstruct.name].dcm_paths.extend(  # type: ignore
                            rtstruct.file_paths
                        )  # type: ignore
                    else:
                        self.rtstruct_dict.add(rtstruct)
            # Close the pool and wait for all the workers to finish
            pool.close()
            pool.join()
            # If output_path is provided, update the JSON file
            if roi_json_path:
                self.rtstruct_dict.write_to_json(roi_json_path)
            b.finish()

    # ---------------------------- New Matching Methods -------------------------
    def calculate_match_score(self, row):
        """
        This method is used to calculate the match score for each roi name using
        the fuzz.ratio method from the fuzzywuzzy package.
        It is possible that a faster better matching score could be obtained. If
        so, this method should be updated.
        """
        roi_lower = row.name.strip()
        max_scores = {}
        for key, val in self.TG263_data_dict.items():
            max_score = 0
            for TG263name in val.synonyms:
                TG263name_lower = TG263name.strip()
                score = fuzz.ratio(
                    TG263name_lower, roi_lower
                )  # replace with function placeholder
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

    ############################################################################
    # -------------- polyfuzz implementation ----------------
    def calculate_match_score_polyfuzz(self):
        # Create a list of all ROI names and a list of all TG263 names
        roi_names = list(self.roi_dict.keys())
        tg263_names = [
            synonym for val in self.TG263_data_dict.values() for synonym in val.synonyms
        ]

        # Create a PolyFuzz instance and match the lists
        model = PolyFuzz("TF-IDF")
        model.match(roi_names, tg263_names)

        # Get the matches
        matches = model.get_matches()

        # Convert the matches DataFrame to a dictionary
        match_dict = matches.set_index("From").T.to_dict("list")

        # Convert the dictionary to a pandas Series
        match_series = pd.Series(match_dict)

        return match_series

    def calculate_match_score_polyfuzz_transformers(self):
        # Create a list of all ROI names and a list of all TG263 names
        roi_names = list(self.roi_dict.keys())
        tg263_names = [
            synonym for val in self.TG263_data_dict.values() for synonym in val.synonyms
        ]

        # Create a TransformerWordEmbeddings instance
        embeddings = TransformerWordEmbeddings("bert-base-multilingual-cased")

        # Create an Embeddings instance and match the lists
        bert = Embeddings(embeddings, min_similarity=0, model_id="BERT")
        model = PolyFuzz(bert)
        model.match(roi_names, tg263_names)

        # Get the matches
        matches = model.get_matches()

        # Convert the matches DataFrame to a dictionary
        match_dict = matches.set_index("From").T.to_dict("list")

        # Convert the dictionary to a pandas Series
        match_series = pd.Series(match_dict)

        return match_series

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

    def prioritize_rt_structures(self):
        """
        Prioritize RT structures for the match process.
        This function returns a sorted list of RT structures, with the most
        important structures first.
        """
        # Define regex patterns for target structures and optimization structures
        target_pattern = re.compile(r"(GTV|CTV|PTV)", re.IGNORECASE)
        optimization_pattern = re.compile(
            r"\bopt(?=[A-Z])|\b(?<=[A-Z])opt\b", re.IGNORECASE
        )

        # Define a list of high priority TG_263_names
        high_priority_names = [
            "Prostate",
            "Bladder",
            "Rectum",
            "Femur_Head_L",
            "Femur_Head_R",
            "Bowel",
            "PenileBulb",
            "SeminalVes",
            "Urethra",
        ]  # Needs to be actual TG_263 names, could replace this to use selected TG_263_data_dict items

        # Initialize an empty list to store the high priority ROIs
        high_priority_rois = []

        # Iterate over the ROIs in the DataFrame
        for roi in self.harmonized_df.index:
            # Skip ROIs that already have a match
            if self.harmonized_df_matches.transpose()[[roi]].any().any() == 1:
                continue

            # Check if the ROI is a target structure and not an optimization structure
            if target_pattern.search(roi) and not optimization_pattern.search(roi):
                high_priority_rois.append(roi)

        for name in high_priority_names:
            top_rows = self.harmonized_df[[name]].nlargest(10, name).index
            for roi in top_rows:
                # Skip ROIs that already have a match
                if self.harmonized_df_matches.transpose()[[roi]].any().any() == 1:
                    continue
                high_priority_rois.append(roi)

        # Sort the high priority ROIs by the number of words and the length of the words
        high_priority_rois = sorted(
            high_priority_rois, key=lambda roi: (len(roi.split("_")), len(roi))
        )

        return high_priority_rois

    def review_matches_gui(self):
        """
        Allows the user to review and confirm the matches for each ROI name,
        and add custom matches if necessary.
        Review is ordered by score, with the highest score first.
        """
        gui = HarmonizerGUI(harmonizer=self)
        gui.run()


################################################################################
# -------------------------- GUI Classes ---------------------------------------
class HarmonizerGUI:
    def __init__(self, harmonizer: StructureHarmonizer):
        self.harmonizer = harmonizer
        self.harmonizer.harmonized_df = self.harmonizer.harmonized_df.transpose()
        self.root = tk.Tk()
        self.root.title("Harmonizer")

        # Create a Label
        self.label = ttk.Label(self.root, text="Please enter a column name:")
        self.label.grid(column=0, row=0)

        # Create a Combobox for input
        self.input_text = ttk.Combobox(
            self.root, values=self.harmonizer.harmonized_df.columns.tolist()
        )
        self.input_text.grid(column=1, row=0)

        # Create a Spinbox to select the number of results
        self.num_results = ttk.Spinbox(self.root, from_=1, to=100)
        self.num_results.grid(column=2, row=0)

        # Create a Text widget to display the results
        self.output = tk.Text(self.root)
        self.output.grid(column=0, row=1, columnspan=3)
        self.num_results = ttk.Spinbox(self.root, from_=1, to=100)
        self.num_results.grid(column=2, row=0)
        self.num_results.delete(0, "end")
        self.num_results.insert(0, "10")  # Set a default value of 10

        # Bind the Combobox and Spinbox to the on_value_change function
        self.input_text.bind("<<ComboboxSelected>>", self.on_value_change)
        self.num_results.bind("<Return>", self.on_value_change)

        # Create a Text widget to display the top matches
        self.output.tag_config("highlight", background="yellow")
        self.output.bind("<Button-1>", self.on_text_click)

    def get_top_rows(self, roi):
        self.top_rows = self.harmonizer.harmonized_df[[roi]].nlargest(
            int(self.num_results.get()), roi
        )

    def on_value_change(self, event):
        # Get the input value
        search_column = self.input_text.get()
        # Check if the input value is a column in the DataFrame
        if search_column in self.harmonizer.harmonized_df.columns:
            # Get the top n rows ordered by score
            self.get_top_rows(search_column)
            # Convert the DataFrame to a string and set it as the value of the output widget
            top_scores_str = self.top_rows.to_string()
            # Use the display_matches method to display the top rows
            self.display_matches(top_scores_str)

    def on_text_click(self, event):
        """
        Handle a mouse button click event on the Text widget.
        """
        # Remove the highlight from all lines
        self.output.tag_remove("highlight", "1.0", tk.END)

        # Get the current line of text under the mouse cursor
        current_line = self.output.get("insert linestart", "insert lineend")

        # Store the selected match
        self.selected_match = current_line.strip()

        # Highlight the selected match
        self.output.tag_add("highlight", "insert linestart", "insert lineend")

    def get_user_selection(self):
        """
        Get the user's selection from the Text widget.
        """
        # Return the selected match
        return self.selected_match

    def get_user_input(self, prompt: str) -> str | None:
        # Create a simple dialog that asks the user for input and returns it
        user_input = simpledialog.askstring("Input", prompt, parent=self.root)
        return user_input

    def display_message(self, message: str):
        # Insert the message at the end of the Text widget and scroll to the end
        self.output.insert(tk.END, message + "\n")
        self.output.see(tk.END)

    def display_matches(self, matches_str):
        """
        Display the given matches in the GUI.
        """
        # Set the string as the value of the output widget
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, matches_str)

    def wait_for_user_input(self):
        """
        Wait for the user to select a match.
        """
        # Create a simple dialog that asks the user for input and returns it
        user_input = simpledialog.askstring(
            "Input", "Please select a match", parent=self.root
        )
        return user_input

    def update_dataframe(self, item, match):
        """
        Update the DataFrame based on the given match.
        """
        # Update the DataFrame
        self.harmonizer.harmonized_df_matches.loc[match, item] = 1
        #  note the reversal of match, item from item, match. This is necessary
        # to account for the transpose of the harmonized_df in the init of the GUI

    def review_matches(self):
        user_responses = {}
        self.harmonizer.user_defined_standard_names = []
        priority_rois = self.harmonizer.prioritize_rt_structures()

        for roi in self.harmonizer.harmonized_df.index:
            if self.harmonizer.harmonized_df.loc[roi].any() == 1:
                continue
            # Get the top N scored items for the current ROI from the GUI
            self.input_text.set(roi)
            self.on_value_change(None)
            top_scores_str = self.output.get(1.0, tk.END)
            # Display the top matches in the GUI
            self.display_matches(top_scores_str)
            # Wait for the user to select the best match
            self.wait_for_user_input()
            # Update the DataFrame based on the user's selection
            selected_match = self.get_user_selection()
            self.update_dataframe(roi, selected_match)

    def run(self):
        self.root.mainloop()

    def run_review(self):
        self.review_matches()
        self.root.mainloop()

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
