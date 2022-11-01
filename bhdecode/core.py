from PIL import Image
import pytesseract
import pandas as pd
import os

from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import pytesseract
import sqlite3
from PIL import Image
from common_functions import (
    extract_text_coordinates,
    extract_page_text,
    get_coordinate_aggregates,
    extract_location,
    get_lat_lon,
    extract_date,
    extract_borehole_number,
)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class BoreLogExtractor:
    """Functions to extract metadata and depth, description fields from borehole logs."""

    def __init__(self):
        self

    def create_metadata_df(self, text: str, file):
        df_city = extract_location(text)
        df_coord = get_lat_lon(text)
        filename = Path(file).stem
        df_bnum = pd.DataFrame([{"Borehole_info": filename}])
        df_date = extract_date(text)
        df_num = extract_borehole_number(text)

        df_output = pd.concat([df_bnum, df_num, df_coord, df_city, df_date], axis=1,)[
            0:1
        ]
        return df_output

    def extract_metadata(self, bfile):
        """Create a dataframe with the extracted metadata from the input file."""

        text = extract_page_text(bfile)

        df = self.create_metadata_df(text, bfile)
        return df

    def extract_depth_values(self, file):
        """Extract the depth values from the images"""

        df = pytesseract.image_to_data(
            Image.open(file), lang="eng", config="--psm 6", output_type="data.frame"
        )

        df = df.loc[~df["text"].isna()]

        df["depth"] = df["text"].str.extract("([0-9]{1,2}[,.][0-9]{1,2})")

        df = df.loc[~df["depth"].isna()]

        df = df[["left", "top", "text"]]
        return df

    def combine_depth_description(self, df1, df2):
        """Combine two dataframes based on a query"""

        # Make the db in memory
        conn = sqlite3.connect(":memory:")
        # write the tables
        df1.to_sql("depth", conn, index=False)
        df2.to_sql("i2d_xy", conn, index=False)
        qry = """
            select  
                Description,left_avg,top_avg,
                depth.text Depth
            from
                i2d_xy left join depth on
                top between top_min and top_max 
            """

        df = pd.read_sql_query(qry, conn)
        return df

    def get_post_process(self, df):
        """Function to clean the depth and description columns"""

        df = df[~df["Depth"].str.contains("[A-Za-z]", na=False)]
        df = df[df["Description"].str.contains("[A-Za-z]", na=False)]
        df = df[~df["Depth"].isna()]
        df = df[df.left_avg > df.left_avg.quantile(0.25)]
        df = df.sort_values(by=["top_avg"])
        df = df[~df["Description"].str.contains("Well|D1:", na=False)]
        df = df[~df["Depth"].str.contains('"')].reset_index(drop=True)
        df = df[["Description", "Depth"]]
        return df

    def extract_depth_desc(self, bfile):
        """Extract the depth and description columns from borehole log. The bounding box cor-ordinate from
        OCRed text is used to create clusters of texts. Then the clusters are positioned based on the boundong
        box co-ordinates."""

        # get coordinates in the dataframe

        depth = self.extract_depth_values(bfile)

        text_coord = extract_text_coordinates(bfile)

        text_coord_xy = text_coord[["top"]]

        ## define clustering parameters
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            affinity="euclidean",
            linkage="ward",
            compute_distances=True,
            distance_threshold=650,
        )
        ## fit clustering model on the 'top' co-ordinate
        indices = clustering_model.fit(text_coord_xy)
        clustering_model.labels_

        text_coord_xy = text_coord[["left", "top"]]
        text_coord_xy["cluster_labels_hac"] = clustering_model.labels_

        text_coord_xy["text"] = text_coord["text"]
        text_coord_xy = text_coord_xy[["left", "top", "text", "cluster_labels_hac"]]

        ## join the words within each of the clusters to create sentences.
        text_coord_xy["Description"] = text_coord_xy.groupby(["cluster_labels_hac"])[
            "text"
        ].transform(lambda x: " ".join(x))

        ## get the min and max of the top and left coordinates, which then can be used for
        ## the position of rows and columns

        text_coord_xy = get_coordinate_aggregates(
            text_coord_xy, "top_min", "cluster_labels_hac", "top", "min"
        )
        text_coord_xy = get_coordinate_aggregates(
            text_coord_xy, "top_max", "cluster_labels_hac", "top", "max"
        )
        text_coord_xy = get_coordinate_aggregates(
            text_coord_xy, "left_avg", "cluster_labels_hac", "left", "mean"
        )
        text_coord_xy = get_coordinate_aggregates(
            text_coord_xy, "top_avg", "cluster_labels_hac", "top", "mean"
        )

        text_coord_xy = text_coord_xy[
            ["left_avg", "top_avg", "top_min", "top_max", "Description"]
        ].drop_duplicates()

        text_coord_xy["Description"] = (
            text_coord_xy["Description"].str.replace(",", ".").str.replace(";", " ")
        )

        ## order by top_avg column
        text_coord_xy = text_coord_xy.sort_values(by=["top_avg"])
        df = self.combine_depth_description(depth, text_coord_xy)
        df = self.get_post_process(df)
        return df
