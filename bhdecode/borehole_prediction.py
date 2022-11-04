import numpy as np
import pandas as pd
from common_functions import transform_data_classify

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


img_size = 224


def get_borelogs(dir, model):
    """Pre-process the images for using in the classification
    model and save the predictions."""

    image_proc = transform_data_classify(dir, img_size)

    x_newval = []
    # y_newval = []
    fname = []

    for feature, name in image_proc:
        # fname=Path(file).stem
        x_newval.append(feature)
        # y_newval.append(label)
        fname.append(name)
        # print(fname)

    # Normalize the data
    x_newval = np.array(x_newval) / 255

    x_newval.reshape(-1, img_size, img_size, 1)
    # y_newval = np.array(y_newval)

    # predictions = model.predict_classes(x_newval)
    predictions = np.argmax(model.predict(x_newval), axis=-1)

    pred = pd.DataFrame(data={"File_name": fname, "Prediction": predictions})
    pred["File_type"] = np.where(pred["Prediction"] == 0, "Borehole", "Non_borehole")
    pred = pred[["File_name", "File_type"]]
    return pred
