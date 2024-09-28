import http.client

# Set the API endpoint and data
url = "localhost"
port = 11434
endpoint = "/api/generate"
data = (
    '{"model": "phi3-3.8b-mini-4k-instruct-fp16", '
    '"prompt": "What is an alternative word for debugging?"}'
)

# Create a connection
conn = http.client.HTTPConnection(url, port)

# Set headers
headers = {
    'Content-type': 'application/json'
}

try:
    # Send the POST request
    conn.request("POST", endpoint, body=data, headers=headers)

    # Get the response
    response = conn.getresponse()
    
    # Ensure the response was successful
    if response.status == 200:
        # Read the response line by line
        for line in response:
            print(line.decode())  # Print the raw line as text
    else:
        print(f"Error: Received status code {response.status}")

finally:
    # Close the connection
    conn.close()
