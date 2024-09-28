import requests

# Set the OpenAI API endpoint and data
url = "http://localhost:11434/v1/completions"  # OpenAI endpoint for completions
headers = {
    "Authorization": "Bearer ollama_api_dummy",  # Replace with your OpenAI API key
    "Content-Type": "application/json"
}
data = {
    "model": "phi3-3.8b-mini-4k-instruct-fp16",  # Specify the model you want to use
    "prompt": "Give me 10 words about debugging.",
    "stream": True  # Enable streaming responses
}

# Send the POST request with streaming enabled
response = requests.post(url, headers=headers, json=data, stream=True)

# Ensure the response was successful
if response.status_code == 200:
    # Iterate over the response line by line
    for line in response.iter_lines():
        if line:  # Filter out empty lines
            print(line.decode())  # Print the raw line as text
else:
    print(f"Error: Received status code {response.status_code}")
