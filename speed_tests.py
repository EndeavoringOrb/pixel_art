import numpy as np
from PIL import Image
from time import perf_counter

test_iters = 10_000
tests = [
    #"asarray",
    #"array",
    "ratio .shape",
    "ratio var"
]

times = []
mean_times = []

if "asarray" in tests:
    print("testing asarray")

    for _ in range(test_iters):
        img = Image.fromarray(np.random.randint(0, 255, (1024,1024,3), dtype=np.uint8))
        start = perf_counter()

        img = np.asarray(img)
        
        end = perf_counter()
        times.append(end-start)

    mean_times.append(np.mean(times))
    print(mean_times[-1])

if "array" in tests:
    print("testing array")

    for _ in range(test_iters):
        img = Image.fromarray(np.random.randint(0, 255, (1024,1024,3), dtype=np.uint8))
        start = perf_counter()

        img = np.array(img)

        end = perf_counter()
        times.append(end-start)

    mean_times.append(np.mean(times))
    print(mean_times[-1])

if "ratio .shape" in tests:
    print("testing ratio .shape")

    for _ in range(test_iters):
        img = np.random.randint(0, 255, (1024,1024,3), dtype=np.uint8)
        start = perf_counter()

        ratio = img.shape[1]/img.shape[0]

        end = perf_counter()
        times.append(end-start)

    mean_times.append(np.mean(times))
    print(mean_times[-1])

if "ratio var" in tests:
    print("testing ratio var")

    for _ in range(test_iters):
        img = np.random.randint(0, 255, (1024,1024,3), dtype=np.uint8)
        start = perf_counter()

        shape = img.shape
        ratio = shape[1]/shape[0]

        end = perf_counter()
        times.append(end-start)

    mean_times.append(np.mean(times))
    print(mean_times[-1])