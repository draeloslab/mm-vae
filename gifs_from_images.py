#This code is to take the umaps and make them into gifs
#There will be 2 gifs for each split, one for the test data and one for the train data
from imageio import v2 as iio
import os


epochs = 5
max_tasks = 10
splits = 5 #start from 0
for i in range(splits):
    for data in ("Test", "Train"):
        frames = []
        for task in range(max_tasks):
                for epoch in range(1, epochs+1):
                    file_path = f"results/default/split{i}/UMAPs/UMAP_task_{task}_epoch_{epoch}_{data}_data.png"
                    if os.path.exists(file_path):
                        frames.append(file_path)

        print(len(frames))
        iio.mimsave(
            f"results/default/gif_for_splits/{data}_data_split_{i}.gif",
            [iio.imread(f) for f in frames],
            duration=[1000]*len(frames),
            loop=0
        )




