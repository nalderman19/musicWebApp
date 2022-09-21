import requests

URL = "http://127.0.0.1/predict"

TEST_FILE_PATH = "Testing_files/sheila_2.wav"


if __name__ == "__main__":
    audio_file = open(TEST_FILE_PATH, "rb")
    values = {"file": (TEST_FILE_PATH, audio_file, "audio/wav")}
    
    response = requests.post(URL, files=values)
    
    data = response.json()
    
    print(f"Predicted Keyword is: {data['keyword']}")