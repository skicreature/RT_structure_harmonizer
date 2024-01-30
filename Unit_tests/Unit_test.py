import unittest
import pytest
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
from fuzzywuzzy import fuzz
from openpyxl import load_workbook  # noqa
from progress.bar import Bar
from pydicom import dcmread

from structure_harmonizer import *
import pathlib

# TODO - actaully finish writing the unit tests and run them to make sure they work


class TestLoadData(unittest.TestCase):
    def setUp(self):
        # set the directory to search
        # directory = '/pct_ids/users/wd982598/test_requite'
        directory = pathlib.Path(
            r"C:\Users\wd982598\Code\Jupyter-sandbox\REQUITE_Prostate\structure_harmonization\test_requite"
        )
        output_file = r"structure_dictionary"
        TG_263_file_path = pathlib.Path(
            r"C:\Users\wd982598\Code\Jupyter-sandbox\REQUITE_Prostate\structure_harmonization\Unit_Test_data\TG263_Nomenclature_Worksheet_20170815.xls"
        )
        # directory = '/mnt/t/Batch_1/'

        # Define the match quality threshold
        threshold = 88

        # Create an instance of StructureHarmonizer and call the harmonize_structures method
        self.harmonizer = StructureHarmonizer(
            directory,
            output_file,
            TG_263_file_path,
            threshold=threshold,
            roi_json_path="Unit_Test_Data\\json_rtstruct_dict_new.json",
        )

        self.harmonizer.load_json_roi_dict(self.harmonizer.roi_json_path)
        self.harmonizer.get_all_TG263names()
        self.harmonizer.load_harmonized_df(
            "Unit_Test_Data\\structure_dictionary_scores.csv"
        )
        self.harmonizer.harmonized_df_matches = self.harmonizer.harmonized_df.applymap(self.harmonizer.match_method)  # type: ignore

    def test_roi_dict_size(self):
        self.setUp()
        roi_dict_size = len(self.harmonizer.roi_dict)
        self.assertEqual(roi_dict_size, 3555)

    def test_TG263_dict_size(self):
        self.setUp()
        TG263_dict_size = len(self.harmonizer.TG263_data_dict)
        self.assertEqual(TG263_dict_size, 717)


if __name__ == "__main__":
    unittest.main()
