import pandas as pd
import urllib
import os
import logging
from multiprocessing import Pool


def read_and_save(filename, save_to, log_to="../data/log/read_and_save.log",
                  amount=None, imtype=".jpg", verbose=False):
    if os.path.exists(log_to):
        os.remove(log_to)
    logging.basicConfig(filename=log_to, level=logging.INFO)

    if verbose:
        logging.info("Reading csv...")

    if amount is None:
        df = pd.read_csv(filename)
        amount = len(df)
        df = pd.read_csv(filename)
    elif type(amount) == int:
        df = pd.read_csv(filename, nrows=amount)
    elif type(amount) == tuple:
        s, f = amount
        df = pd.read_csv(filename)[s:f]
        amount = f - s
    else:
        raise TypeError("Invalid amount value: {}".format(type(amount)))

    if verbose:
        logging.info("Done, {} url's collected".format(amount))

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    if not imtype.startswith("."):
        imtype = "." + imtype

    pool = Pool()  # no arguments = max pools awailable
    for idx, row in df.iterrows():
        url = row['url']
        name = row['id'] + imtype
        pool.apply_async(save_single, args=(url, save_to, name))
        # save_single()

        if verbose and idx % 100 == 99:
            logging.info("{} / {} images downloaded".format(idx + 1, amount))

    pool.close()
    pool.join()


def save_single(url, dest, name):
    try:
        with urllib.request.urlopen(url) as r:
            with open(os.path.join(dest, name), 'wb') as f:
                # save file
                f.write(r.read())
    except urllib.error.URLError as e:
        logging.error("{} cannot be downloaded".format(url))


if __name__ == "__main__":
    read_and_save("../data/csv/index.csv", "../data/index", verbose=True, log_to="../data/log/index.log")
    read_and_save("../data/csv/test.csv", "../data/test", verbose=True, log_to="../data/log/test.log")
    # url = "https://lh3.googleusercontent.com/-_Iy5pcnRsiY/TlH629GKcZI/AAAAAAAAGC4/iLwQqLxrXGg/s1600/"
    # print(url)
