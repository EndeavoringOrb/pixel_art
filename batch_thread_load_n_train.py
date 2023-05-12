print("importing modules...")
import train
import pandas as pd
import requests
from PIL import Image
import io
import numpy as np
import sentence_similarity as ss
from skimage.transform import resize
from skimage import io as ski_io
from time import perf_counter, localtime, time, sleep
import os
import threading
import queue
from requests.adapters import HTTPAdapter
import warnings
from tensorflow import saved_model
from crop import center_crop

# edit these vars
show_display = False # this does not currently work with the threading
save_interval = 60 # how often to save the model (in seconds)
train_queue_max = 10
row_queue_max = 100
batch_size = 30
text_embedding_type = "clip" # 'clip', 'bert', or 'mini'
download_thread_number = batch_size
sliding_window_size = 30 # how many batches to average the sliding window score over
degrade_iters = 11
crop_imgs = True # do not set this to true if use_img_ratio is True
use_img_ratio = False
load_model = True
load_model_path = "latest" # set to relative path if you want a specific model, otherwise set to latest to load the latest model
#######################################
# dont edit anything below here
assert not (crop_imgs and use_img_ratio), "cannot have crop_imgs and use_img_ratio both be True."
assert text_embedding_type in ['clip', 'bert', 'mini'], "text_embedding_type must be either 'clip', 'bert', or 'mini'."

losses = []

if load_model_path == "latest" and load_model:
    if use_img_ratio:
        load_model_path = ""
        load_model_paths = [i.split("\\")[0] for i in os.listdir("models")]

        for i in range(len(load_model_paths)-1, -1, -1):
            if load_model_paths[i][-2:] == "ir":
                thing = load_model_paths[i][-7:-3]
                if load_model_paths[i][-7:-3] == text_embedding_type:
                    load_model_path = load_model_paths[i]

        assert load_model_path!="", "no compatible model found. you should set load_model to False."
    elif crop_imgs:
        load_model_path = ""
        load_model_paths = [i.split("\\")[0] for i in os.listdir("models")]

        for i in range(len(load_model_paths)-1, -1, -1):
            if load_model_paths[i][-2:] == "cr":
                thing = load_model_paths[i][-7:-3]
                if load_model_paths[i][-7:-3] == text_embedding_type:
                    load_model_path = load_model_paths[i]

        assert load_model_path!="", "no compatible model found. you should set load_model to False."
    else:
        load_model_path = ""
        load_model_paths = [i.split("\\")[0] for i in os.listdir("models")]

        for i in range(len(load_model_paths)-1, -1, -1):
            if load_model_paths[i][-7:-3] == text_embedding_type:
                    load_model_path = load_model_paths[i]
        
        assert load_model_path!="", "no compatible model found. you should set load_model to False."
elif load_model:
    load_model_path = load_model_path.split("\\")[1]
if load_model:
    print(f"loading saved model from {load_model_path}")
    train.model = saved_model.load("models/"+load_model_path)

# disable warnings from urllib
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Create a connection pool
adapter = HTTPAdapter(pool_connections=download_thread_number,pool_maxsize=download_thread_number,max_retries=0)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

# create a queue to store the rows that need to be processed
print("creating queues")
row_queue = queue.Queue()
train_queue = queue.Queue()

# initialize training step count variable
if load_model:
    loaded_row = int(load_model_path.split("_")[1])
    train_step_count = int(load_model_path.split("_")[3].split("-")[0])
else:
    loaded_row = -1
    train_step_count = 0

def download_images():
    global total_index
    while True:
        while train_queue.qsize() > train_queue_max:
            sleep(0.1)
        row = row_queue.get()
        total_index += 1
        if row is None:
            break
        url = row['URL']
        try:
            response = http.get(url)
            # Read image from io.BytesIO
            img = np.asarray(Image.open(io.BytesIO(response.content),mode="r"))
            img = img[:,:,:3]
            if img.dtype == np.uint8:
                img = img / 255
        except Exception as e:
            row_queue.task_done()
            #thinsdg = e.args[0]
            #if e.args[0] == 'too many indices for array: array is 2-dimensional, but 3 were indexed':
            #    img = np.repeat(img, 3, axis=2)
            #    img = np.clip(img[:,:,:3], 0, 1)
            #else:
            continue

        if text_embedding_type == "bert":
            text_embedding = ss.get_bert_embedding(row['TEXT'])
        elif text_embedding_type == "mini":
            text_embedding = ss.get_embeddings(row['TEXT'])
        else:
            text_embedding = ss.get_clip_embeddings(row['TEXT'])
        
        shape = img.shape
        ratio = shape[1]/shape[0]
        if crop_imgs:
            img = center_crop(img)
        img = resize(img,(32,32,3))
        row_queue.task_done()
        for _ in range(degrade_iters):
            degraded_img = train.add_noise(img)
            # add the downloaded image and text embedding to the training queue
            train_queue.put((np.array([img]), np.array([degraded_img]), np.array(text_embedding).reshape((1,-1,1)), ratio))
            img = train.add_noise(img)

def train_model():
    global train_step_count
    global last_elapsed
    global start
    global losses
    while True:
        # get next item
        load_start = perf_counter()
        items = [train_queue.get() for _ in range(batch_size)]
        load_end = perf_counter()
        time_elapsed = load_end-load_start
        for item in items:
            if item is None:
                break
        imgs = [item[0][0] for item in items]
        degraded_imgs = [item[1][0] for item in items]
        text_embeddings = [item[2][0] for item in items]
        ratios = [item[3] for item in items]
        imgs, degraded_imgs, text_embeddings, ratios = np.array(imgs, dtype=np.float32), np.array(degraded_imgs, dtype=np.float32), np.array(text_embeddings, dtype=np.float32), np.array(ratios, dtype=np.float32)


        print(f"time to get {batch_size} items: {time_elapsed:.5f} - rows used: {total_index} - approx. row queue len: {row_queue.qsize()} - approx. train queue len: {train_queue.qsize()} - training step {train_step_count} - ",end="")
        if load_model:
            loss = train.loaded_batch_train_step(imgs,degraded_imgs,text_embeddings,ratios)
        else:
            loss = train.batch_train_step(imgs,degraded_imgs,text_embeddings,ratios)
        loss = float(str(loss).split("(")[1].split(",")[0])
        losses.append(loss)
        losses = losses[-sliding_window_size:]
        print(f"loss: {loss:.5f} - last {sliding_window_size} batch loss: {sum(losses)/len(losses):.5f}",end="\r")

        print()
        train_step_count += batch_size
        train_queue.task_done()

        elapsed = perf_counter() - start
        if elapsed > last_elapsed + save_interval:
            last_elapsed = elapsed
            timestamp = localtime(time())
            timestamp = f"Y{timestamp.tm_year}M{timestamp.tm_mon}D{timestamp.tm_mday}H{timestamp.tm_hour}M{timestamp.tm_min}S{timestamp.tm_sec}"
            saved_model.save(train.model, f"models/row_{total_index}_step_{train_step_count}-{timestamp}_{text_embedding_type}{'_ir' if use_img_ratio else ''}{'_cr' if crop_imgs else ''}")
            #train.model.save(f"models/row_{train_step_count}-{timestamp}")

# start training counter
start = perf_counter()
last_elapsed = 0

# create the threads
print("creating download threads...")
download_threads = []
for i in range(download_thread_number): # create download threads
    t = threading.Thread(target=download_images)
    t.daemon = True
    t.start()
    download_threads.append(t)

print("creating train threads...")
train_threads = []
for i in range(1): # create training threads
    t = threading.Thread(target=train_model)
    t.daemon = True
    t.start()
    train_threads.append(t)

total_index = 0
read_indxs = 0

# loop through the dataframe
for filename in os.listdir("data/archive"):
    # Load the Parquet file
    print(f"reading file data/archive/{filename}...")
    df = pd.read_parquet(f'data/archive/{filename}')
    print("finished reading file.")

    # add rows to the queue
    read_indxs += df.size
    if total_index < read_indxs:
        print(f"putting {df.size} dataframe rows in row_queue...")
        for index, row in df.iterrows():
            while row_queue.qsize() > row_queue_max:
                sleep(0.1)
            if total_index > loaded_row:
                row_queue.put(row)
            else:
                total_index += 1
        

    # wait for all rows to be downloaded
    print("waiting for row_queue.join()")
    row_queue.join()

    # wait for all training to be completed
    print("waiting for train_queue.join()")
    train_queue.join()

# stop the threads
for thread in download_threads:
    thread.join()
for thread in train_threads:
    thread.join()