import requests

def class_test_main():
    url = "http://localhost:8000/health"
    response = requests.get(url)
    print(response)
    try: 
        print("Health check response:", response.json())
    except Exception as e:
        print("Failed to parse JSON response:", e)
        print("Raw response text:", response.text)


def test_classify_malignant_sample():
    mal_sample_file = "example_inputs/malignant_sample.json"
    with open(mal_sample_file, "r") as f:
        mal_sample = f.read()   
    url = "http://localhost:8000/predict"
    response = requests.post(url, data=mal_sample, headers={"Content-Type": "application/json"})
    try:
        print("Malignant sample prediction response:", response.json())
    except Exception as e:
        print("Failed to parse JSON response:", e)
        print("Raw response text:", response.text)  


if __name__ == "__main__":
    #class_test_main()
    test_classify_malignant_sample()