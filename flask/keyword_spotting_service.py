import tensorflow.keras as keras
import numpy as np
import librosa as lbs

MODEL_PATH = "model.h5"
NUM_SAMPLES = 22050

class _Keyword_Spotting_Service:
    # singleton - only one instance
    
# class variables
    model = None
    _mappings = [
        "right",
        "go",
        "no",
        "left",
        "stop",
        "sheila",
        "zero",
        "up",
        "down",
        "yes",
        "on",
        "off"]
    _instance = None
    
    
    def predict(self, file_path):
        # extract MFCCC
        mfccs_bad_shape = self.preprocess(file_path) #     (# segments, # coefficients, 1))
        
        # convert 2d MFCC into 4d array -> (# samples, # segments, # coefficients, # channels)
        mfccs = mfccs_bad_shape[np.newaxis, ...,np.newaxis]
        
        # make a prediction
        predictions = self.model.predict(mfccs)
        predicted_keyword = self._mappings[np.argmax(predictions)]
        
        return predicted_keyword
        
    
    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        
        # load audio file
        signal, sr = lbs.load(file_path)
        
        # ensure concsistency in the audio file length
        if len(signal) >= NUM_SAMPLES:
            signal = signal[:NUM_SAMPLES]
        else: # signal has too few sample points
            signal.resize((22050), refcheck=False)
            
        
        # extract mfcc
        mfccs = lbs.feature.mfcc(signal, sr=NUM_SAMPLES, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        return mfccs.T
    
def Keyword_Spotting_Service():
    # ensure there is only one instance of kss
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
        
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":
    
    kss = Keyword_Spotting_Service()
    
    keyword1 = kss.predict("Testing_files/down_1.wav")
    keyword2 = kss.predict("Testing_files/go_2.wav")
    keyword3 = kss.predict("Testing_files/sheila_1.wav")
        
    






