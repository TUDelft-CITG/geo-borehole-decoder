import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bhdecode",
    )
)
from borehole_prediction import get_borelogs
from common_functions import convert_pdf_image, check_dir_create
from core import BoreLogExtractor
from pathlib import Path
import pandas as pd
import keras

pd.options.mode.chained_assignment = None

## locations of inputs,intermediate and output folders
current = Path.cwd()
## location if the pdf files
input = current / "input_pdf"
## location of the pdfs converted to images
input_images = check_dir_create(current, "pdf_images")
## location of output files
output = check_dir_create(current, "output")

## get the borehole log extractor module
be = BoreLogExtractor()

## load the trained model
model = keras.models.load_model("model.h5")

##convert input pdfs to images
convert_pdf_image(input, input_images)

## classify the inputs to borehole logs : yes/no
bore_pred = get_borelogs(input_images, model)
bores = bore_pred.loc[bore_pred["File_type"] == "Borehole"]

## if the page is borehole log, do the extractions
for file in os.listdir(input_images):
    filename = Path(file).stem

    if filename in bores["File_name"].values:
        metadata = be.extract_metadata(input_images / file)

        content = be.extract_depth_desc(input_images / file)

        # save the output to excel file
        with pd.ExcelWriter(rf"{output}/{filename}.xlsx") as writer:
            metadata.to_excel(writer, sheet_name="metadata", index=False)
            content.to_excel(writer, sheet_name="depth_description", index=False)

    else:
        print(rf"{file} is not a borehole log.")
