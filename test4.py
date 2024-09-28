import http.client

# Set the OpenAI API endpoint and data
url = "localhost"
port = 11434
endpoint = "/v1/completions"
data = (
    '{"model": "phi3-3.8b-mini-4k-instruct-fp16", '
    '"prompt": "What is an alternative word for debugging?", '
    '"stream": true}'  # Enable streaming responses
)

# Create a connection
conn = http.client.HTTPConnection(url, port)

# Set headers
headers = {
    "Authorization": "Bearer ollama_api_dummy",  # Replace with your OpenAI API key
    "Content-Type": "application/json"
}

try:
    # Send the POST request
    conn.request("POST", endpoint, body=data, headers=headers)

    # Get the response
    response = conn.getresponse()
    
    # Ensure the response was successful
    if response.status == 200:
        # Read the response line by line
        while True:
            line = response.readline()  # Read line by line
            if not line:  # Break the loop if no more lines
                break
            print(line.decode())  # Print the raw line as text
    else:
        print(f"Error: Received status code {response.status}")

finally:
    # Close the connection
    conn.close()
