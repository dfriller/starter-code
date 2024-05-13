import requests
import json

# Define the API URL
api_url = "https://dustin-udacity-project-14a2f087c6c6.herokuapp.com/inference/"

# Prepare the sample payload
sample_payload = {
    'age': 50,
    'workclass': "Private",
    'fnlgt': 234721,
    'education': "Doctorate",
    'education_num': 16,
    'marital_status': "Separated",
    'occupation': "Exec-managerial",
    'relationship': "Not-in-family",
    'race': "Black",
    'sex': "Female",
    'capital_gain': 0,
    'capital_loss': 0,
    'hours_per_week': 50,
    'native_country': "United-States"
}

# Convert the payload to a JSON string
json_payload = json.dumps(sample_payload)

# Send the POST request
response = requests.post(api_url, data=json_payload, headers={"Content-Type": "application/json"})

# Check the response status code
if response.status_code == 200:
    print("Success!")
    print("Response JSON:", response.json())
else:
    print(f"Failed with status code: {response.status_code}")
    print("Response Text:", response.text)
