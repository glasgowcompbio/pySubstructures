import gzip
import os
import pathlib
import pickle
import requests
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm

from loguru import logger


class MS1(object):
    def __init__(
        self,
        id,
        mz,
        rt,
        intensity,
        file_name,
        scan_number=None,
        single_charge_precursor_mass=None,
    ):
        self.id = id
        self.mz = mz
        self.rt = rt
        self.intensity = intensity
        self.file_name = file_name
        self.scan_number = scan_number
        if single_charge_precursor_mass:
            self.single_charge_precursor_mass = single_charge_precursor_mass
        else:
            self.single_charge_precursor_mass = self.mz
        self.name = "{}_{}".format(self.mz, self.rt)

    def __str__(self):
        return self.name


########################################################################################################################
# Common methods
########################################################################################################################


def create_if_not_exist(out_dir):
    if not os.path.exists(out_dir) and len(out_dir) > 0:
        logger.info("Created %s" % out_dir)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)


def save_obj(obj, filename):
    """
    Save object to file
    :param obj: the object to save
    :param filename: the output file
    :return: None
    """

    # workaround for
    # TypeError: can't pickle _thread.lock objects
    # when trying to pickle a progress bar
    if hasattr(obj, "bar"):
        obj.bar = None

    out_dir = os.path.dirname(filename)
    create_if_not_exist(out_dir)
    logger.info("Saving %s to %s" % (type(obj), filename))
    with gzip.GzipFile(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    """
    Load saved object from file
    :param filename: The file to load
    :return: the loaded object
    """
    try:
        with gzip.GzipFile(filename, "rb") as f:
            return pickle.load(f)
    except OSError:
        logger.warning(
            "Old, invalid or missing pickle in %s. Please regenerate this file."
            % filename
        )
        return None


def download_example_data():
    # Check if the data already exists
    expected_file = "example_data/Beer_multibeers_1_T10_POS.mzML"  # Adjust as necessary for your data
    if os.path.exists(expected_file):
        logger.info(f"Data file {expected_file} already exists. Skipping download.")
        return

    # Proceed with download if the data does not exist
    url = "https://github.com/glasgowcompbio/vimms-data/raw/main/example_data.zip"
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    logger.info("Downloading example data.")
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    with BytesIO() as zip_file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            zip_file.write(data)
        t.close()

        zip_file.seek(0)
        with ZipFile(zip_file) as z:
            z.extractall(".")
    logger.info("Download and extraction complete.")
