# obtain mfcc of each audio file and store in json file

import librosa as lbs
import json
import os

DATASET_PATH = "dataset/"
JSON_PATH = "speech_data.json"

SAMPLE_RATE = 22050

def prepare_dataset(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, n_segments=5): 
    # number of segments chops up each track into segments to increase number of input data
    
    # build dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
        "files": []
    }

    
    for i, (dir_path, dir_names, file_names) in enumerate(os.walk(dataset_path)):
        
        if dir_path is not dataset_path:
            # split file into components
            mapping = dir_path.split("/")[-1]
            data["mapping"].append(mapping)
            print(f"Processing word:   {mapping}")
            
            for f in file_names:
                # now get mfccs for each file in semantic label folder
                file_path = os.path.join(dir_path, f)
                signal, sr = lbs.load(file_path, sr=SAMPLE_RATE)
                
                
                if len(signal) >= SAMPLE_RATE:
                    mfcc = lbs.feature.mfcc(signal[:SAMPLE_RATE],
                                            sr=sr,
                                            n_fft=n_fft,
                                            n_mfcc=n_mfcc,
                                            hop_length=hop_length)
                
                
                    data["mfcc"].append(mfcc.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print(f"{file_path}: {i-1}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    
    
    
if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)