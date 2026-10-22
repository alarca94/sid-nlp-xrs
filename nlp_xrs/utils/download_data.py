import argparse
import os

import requests
from tqdm import tqdm


def fetch_or_resume(url, out):
    with open(out, "ab") as f:
        headers = {}
        pos = f.tell()
        if pos:
            headers["Range"] = f"bytes={pos}-"
        response = requests.get(url, headers=headers, stream=True)
        total_size = int(response.headers.get("content-length"))
        f.writelines(tqdm(iterable=response.iter_content(chunk_size=1024), total=total_size // 1024, unit="KB"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="amazon", help="name of dataset")
    parser.add_argument("--category", "-c", type=str, default="Automotive", help="name of dataset")

    args, _ = parser.parse_known_args()
    data_path = "English-Jar"

    if args.dataset.lower() == "amazon":
        # Old URL: wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games.json.gz
        base_url = "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/"
        prefix = ""
        suffix_url = "_5.json.gz"
        output_dir = "data/amazon/compressed"

        filename = prefix + args.category.replace(" ", "_") + suffix_url
        if not os.path.exists(os.path.join(output_dir, filename)):
            print(f"\nDownloading {args.category} subset from Amazon dataset...")
            os.system(f"wget {base_url + filename} -P {output_dir}")
        else:
            print(f"\n{args.category} subset from Amazon dataset already exists...")

    else:
        raise NotImplementedError("The download of the provided dataset is not supported yet")
