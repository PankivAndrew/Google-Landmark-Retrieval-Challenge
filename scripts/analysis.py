import pandas as pd
from PIL import Image
import os

def analyzer(path_to_images='../data/images', save_to="../data/info.csv"):
    filenames = os.listdir(path_to_images)
    sizes = {}
    for filename in filenames:
        image = Image.open(os.path.join(path_to_images, filename))
        sizes[image.size] = sizes.get(image.size, 0) + 1
    sizes = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(sizes, columns=["size", "count"])
    df.to_csv(save_to)

if __name__ == "__main__":
    analyzer()
