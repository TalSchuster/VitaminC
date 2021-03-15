import os
import urllib.request as request
import zipfile
import logging


logger = logging.getLogger(__name__)


TASK2PATH = {
    "vitaminc" : "https://www.dropbox.com/s/ivxojzw37ob4nee/vitaminc.zip?dl=1",
    "vitaminc_real" : "https://www.dropbox.com/s/y74sez0qa8z5dnw/vitaminc_real.zip?dl=1",
    "vitaminc_synthetic" : "https://www.dropbox.com/s/uqb2316chhtx77z/vitaminc_synthetic.zip?dl=1",
}


def download_and_extract(task, data_dir):
    if task not in TASK2PATH:
        logger.warning("No stored url for task %s. Please download manually." % task)
        return
    logger.info("Downloading and extracting %s..." % task)
    data_file = "%s.zip" % task
    request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    logger.info("Completed! Stored at %s" % data_dir)
