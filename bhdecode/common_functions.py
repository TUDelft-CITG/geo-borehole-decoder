from importlib.resources import path
from pathlib import Path
import pandas as pd
import re
import geonamescache
import spacy
import numpy as np
from pdf2image import convert_from_path
from pytesseract import pytesseract
from PIL import Image
import glob
import os
import cv2
from keras.models import load_model
import h5py

from io import BytesIO

pd.options.mode.chained_assignment = None

pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def convert_pdf_image(input, output):
    """Convert each page of the pdf files in the input folder to image files in 'jpeg' format and save in the ouput folder."""
    pdf_files = glob.glob(os.path.join(input, "*.pdf"))
    for pdf_file in pdf_files:
        pdf_name = Path(pdf_file).stem
        pages = convert_from_path(pdf_file, 500)
        ## save the pages
        for page_no, image in enumerate(pages):
            image.save(rf"{output}/{pdf_name}-{page_no}.jpeg")


def extract_date(text: str):
    """Extract dates from text."""
    dates = []
    date_exps = ["\d+/\d+/\d+", "\d+-\d+-\d+"]

    for d in date_exps:
        dt = re.search(d, text)
        if dt:
            dates.append(dt.group(0))

    df_date = pd.DataFrame()
    df_date["Date"] = dates
    return df_date


def get_lat_lon(text: str):
    """Extract latitude and logitude from text. Use patterns to
    identify latitude/logitudes."""
    lat_l = list()
    lon_l = list()
    lat_exps = [
        "[N]\d{7}\,\d{2}?",
        "[N]\s?\d{7}\s?",
        "\d{5}\s?m",
        "\d{5}\.\d{2}",
        "\d{5}",
        "\d{2}°\s?\d{2}'\s?\d{2}\.\d{1}\"[Ss]",
    ]
    lon_exps = [
        "[E]\d{6}\s?\,\d{2}?",
        "[E]\s?\d{6}\s?",
        "\d{6}",
        "\d{6}\s?m",
        "\d{3}°\s?\d{2}'\s?\d{2}\.\d{1}\"\s[Ee]",
    ]

    for lat in lat_exps:
        result = re.search(lat, text)
        if result:
            lat_l.append(result.group(0))

    for lon in lon_exps:
        lon = re.search(lon, text)
        if lon:
            lon_l.append(lon.group(0))

    df_cord = pd.DataFrame()

    df_cord = pd.DataFrame.from_dict(
        {"Latitude/Northing/Y": lat_l, "Longitude/Easting/X": lon_l}, orient="index"
    ).T
    return df_cord


def extract_page_text(file):
    """Extract the contents of entire page into a dataframe."""
    extracted_text = []
    picture = Image.open(file)
    text = pytesseract.image_to_string(picture, lang="eng")
    extracted_text.append(text)
    df = pd.DataFrame(extracted_text, columns=["Page_Text"]).replace(
        r"\n", " ", regex=True
    )
    text = df["Page_Text"][0]
    return text


def extract_dict_value(var, key):
    """Function to search for a key in a nested Python dict and return value"""
    if isinstance(var, dict):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, (dict, list)):
                yield from extract_dict_value(v, key)
    elif isinstance(var, list):
        for d in var:
            yield from extract_dict_value(d, key)


def extract_location(text: str):
    """Function for extracting named entities such as city and
    country using spaCy model and save in a dataframe"""

    gc = geonamescache.GeonamesCache()
    # gets nested dictionary for countries
    countries = gc.get_countries()
    # gets nested dictionary for cities
    cities = gc.get_cities()
    cities = [*extract_dict_value(cities, "name")]
    countries = [*extract_dict_value(countries, "name")]

    ## use spacy's trained model for tagging the locations.
    nlp = spacy.load("en_core_web_lg")
    city_name = list()
    state_name = list()
    country_name = list()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            if ent.text in countries:
                country_name.append(ent.text)
            elif ent.text in cities:
                city_name.append(ent.text)
            else:
                state_name.append(ent.text)

    df_location = pd.DataFrame.from_dict(
        {"City": city_name, "State": state_name, "Country": country_name},
        orient="index",
    ).T
    return df_location


def listing_splitter(text, listing):
    """Function which takes a string and a list of words to extract as inputs."""
    # Try except to handle np.nans in input
    try:
        # Extract the list of flags
        flags = [l for l in listing if l in text.lower()]
        # If any flags were extracted then return the list
        if flags:
            return flags
        # Otherwise return np.nan
        else:
            return np.nan
    except AttributeError:
        return np.nan


def extract_borehole_number(text: str):
    """Extract borehole number from text."""
    num = []
    boreexps = ["B-\d+", "BH-\d+", "BH_[A-Za-z]*\d+", "EH\d+"]

    for bnum in boreexps:
        result = re.search(bnum, text)
        if result:
            num.append(result.group(0))

    df_bore = pd.DataFrame()
    df_bore["Bore_num"] = num
    return df_bore


def extract_text_coordinates(file):
    """Extract text and bounding box coordinates if the
    extracted text frim the image file"""
    df = pytesseract.image_to_data(
        Image.open(file), lang="eng", output_type="data.frame"
    )
    df = df.loc[~df["text"].isna()]
    return df


def get_coordinate_aggregates(df, new_col, group_col, value_col, fun):
    """Create aggreagtes"""
    df[new_col] = (df).groupby([group_col])[value_col].transform(fun)
    return df


def transform_data_classify(data_dir, img_size):
    """Pre-process the images"""
    data = []

    for img in os.listdir(data_dir):
        fname = Path(img).stem
        try:
            img_arr = cv2.imread(os.path.join(data_dir, img))[
                ..., ::-1
            ]  # convert BGR to RGB format
            resized_arr = cv2.resize(
                img_arr, (img_size, img_size)
            )  # Reshaping images to preferred size
            data.append([resized_arr, fname])
        except Exception as e:
            print(e)
    return np.array(data)


def check_dir_create(root, new):
    """Check if a directory exists in the root directory, if not create it and return the path"""
    n_path = root / new

    if not os.path.exists(n_path):
        os.makedirs(n_path)
    return n_path
