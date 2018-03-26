import os
import csv


def get_data_and_labels(path):
    data = []
    labels = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            photos, names = zip(*[[[photo, name] for photo in files] for root, dirs, files in
                                  os.walk(os.path.join(root, name))][0])
            data += photos
            labels += names
    return data, labels


def into_csv(csv_name, data, labels):
    with open(csv_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(data, labels))


if __name__ == "__main__":
    data, labels = get_data_and_labels('../data/paris_data-set/')
    into_csv('../data/csv/data.csv', data, labels)
