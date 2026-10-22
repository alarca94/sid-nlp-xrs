import argparse
import gzip
import os

import pandas as pd

from nlp_xrs.utils.in_out import get_paths

AMAZON_DOMAINS = (
    "Amazon_Instant_Video",
    "Apps_for_Android",
    "Automotive",
    "Baby",
    "Beauty",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Health_and_Personal_Care",
    "Grocery_and_Gourmet_Food",
    "Home_and_Kitchen",
    "Kindle_Store",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Sports_and_Outdoors",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
)

AMAZON_URL_PREFIX = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"


def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def download_amazon(domain: str, data_path: str):
    if domain not in AMAZON_DOMAINS:
        raise ValueError(f"Domain {domain} is not supported")

    os.makedirs(os.path.join(data_path, "orig"), exist_ok=True)
    reviews_path = os.path.join(data_path, "orig", "reviews.json.gz")
    meta_path = os.path.join(data_path, "orig", "meta.json.gz")

    if not os.path.exists(reviews_path):
        print(f"Downloading {reviews_path}...")
        os.system(f"curl -o {reviews_path} {AMAZON_URL_PREFIX + f'reviews_{domain}.json.gz'}")

    if not os.path.exists(meta_path):
        print(f"Downloading {meta_path}...")
        os.system(f"curl -o {meta_path} {AMAZON_URL_PREFIX + f'meta_{domain}.json.gz'}")

        # Make sure the meta file is readable by pandas read_json
        meta_df = getDF(meta_path)
        meta_df.to_json(meta_path, orient="records", lines=True, compression="gzip")


def download_datasets(dataset_names: list[str], base_path: str):
    for dataset_name in dataset_names:
        if dataset_name.startswith("Amazon_"):
            domain = dataset_name.split("_", 1)[1]
            data_path = os.path.dirname(get_paths(domain, "0", base_path).data)
            download_amazon(domain=domain, data_path=data_path)
        # elif dataset_name == "RateBeer":
        #     download_ratebeer()
        # elif dataset_name == "TripAdvisor":
        #     download_tripadvisor()
        # elif dataset_name == "Yelp":
        #     download_yelp()
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported yet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-names", type=lambda x: x.split(","), default="Amazon_Beauty")
    parser.add_argument("--base-path", type=str, default=os.path.abspath(os.path.dirname(__file__)))
    args = parser.parse_args()

    download_datasets(dataset_names=args.dataset_names, base_path=args.base_path)
