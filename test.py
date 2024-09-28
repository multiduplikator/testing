import requests

# Set the API endpoint and data
url = "http://localhost:11434/api/generate"
data = {
    "model": "phi3-3.8b-mini-4k-instruct-fp16",
    "prompt": "Give me 10 words about debugging."
}

# Send the POST request with streaming enabled
response = requests.post(url, json=data, stream=True)

# Ensure the response was successful
if response.status_code == 200:
    # Iterate over the response line by line
    for line in response.iter_lines():
        if line:  # Filter out empty lines
            print(line.decode())  # Print the raw line as text
else:
    print(f"Error: Received status code {response.status_code}")
