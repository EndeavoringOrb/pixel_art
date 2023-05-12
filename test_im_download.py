import concurrent.futures
import os
import requests
import pandas as pd
from time import perf_counter
from PIL import Image
import io

def save_image_from_url(url, output_folder):
    global total_files_downloaded
    global time_start
    image = requests.get(url['URL'])
    img = Image.open(io.BytesIO(image.content))
    total_files_downloaded += 1
    print(f"files: {total_files_downloaded} - files per second: {total_files_downloaded/(perf_counter()-time_start):.2f}")
    #output_path = os.path.join(
    #    output_folder, url.image_name
    #)
    #with open(output_path, "wb") as f:
    #    f.write(image.content)

def load(df, workers, output_folder):    
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        future_to_url = {
            executor.submit(save_image_from_url, url, output_folder): url for url in df.iterrows()
        }
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                future.result()
            except Exception as exc:
                print(
                    "%r generated an exception: %s" % (url, exc)
                )

total_files_downloaded = 0

# loop through the dataframe
for filename in os.listdir("data/archive"):
    # Load the Parquet file
    print(f"reading file data/archive/{filename}...")
    df = pd.read_parquet(f'data/archive/{filename}')
    print("finished reading file.")

    time_start = perf_counter()
    load(df, 5, "")

