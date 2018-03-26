import os

from PIL import Image
import logging
from multiprocessing import Pool
from PIL.ExifTags import TAGS, GPSTAGS


def resize_images(in_folder, out_folder, size=(640, 480), verbose=False, log_to="../data/log/preprocessing.log"):
    in_files = os.listdir(in_folder)

    logging.basicConfig(filename=log_to, level=logging.INFO)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    pool = Pool()  # no arguments = max pools awailable
    for i, filename in enumerate(in_files):
        pool.apply_async(resize_single, args=(filename, in_folder, out_folder, size))
        if verbose and i % 100 == 99:
            logging.info("{} / {} images resized".format(i + 1, len(in_files)))

    pool.close()
    pool.join()


def resize_single(filename, in_folder, dest, size=(640, 480)):
    im = Image.open(os.path.join(in_folder, filename)).resize(size)
    im.convert('RGB').save(os.path.join(dest, filename))

if __name__ == "__main__":
    resize_images("../data/index", "../data/index_resized", verbose=True, size=(160, 120))
    resize_images("../data/test", "../data/test_resized", verbose=True, size=(160, 120))
    # count_gps_info("../data/index")
