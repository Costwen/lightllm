import subprocess
import time
import requests
import json
import base64
from model_infer import test_model_inference

def history_to_prompt(signal_type, history, query):
    if signal_type == 'base':
        return query
    elif signal_type == 'vqa':
        answer_format = 'Short answer:'
    elif signal_type == 'chat':
        answer_format = 'Answer:'
    else:
        assert False, f"Unknown signal type {signal_type}"

    prompt = ''
    for i, (old_query, response) in enumerate(history):
        prompt += 'Question: ' + old_query + " {} ".format(answer_format) + response + "\n"
    prompt += 'Question: {} {}'.format(query, answer_format)
    return prompt

def start_server():
    # Command to start the server
    command = [
        'python', '-m', 'lightllm.server.api_server',
        '--host', '0.0.0.0',
        '--port', '8080',
        '--tp', '1',
        '--max_total_token_num', '4096',
        '--trust_remote_code',
        '--enable_multimodal',
        '--cache_capacity', '1000',
        '--model_dir', 'pretrain/cogvlm'  # Replace with correct path
    ]
    output_file = "log"
    with open(output_file, "w") as fout:
        subprocess.Popen(command, stdout=fout)

def main():
    # Start the server
    server_process = start_server()

    try:
        # Wait for the server to start
        time.sleep(200)  # Adjust the sleep time if necessary

        # Your client code
        url = 'http://localhost:8080/generate'
        headers = {'Content-Type': 'application/json'}

        uri = "1.png" # Replace with the correct path
        if uri.startswith("http"):
            images = [{"type": "url", "data": uri}]
        else:
            with open(uri, 'rb') as fin:
                b64 = base64.b64encode(fin.read()).decode("utf-8")
            images = [{'type': "base64", "data": b64}]

        query = "Tell the ads image content and the text in the image., Response with the format: CONTENT: [rich image content] \n TEXT:[image text]"
        query = history_to_prompt("chat", [], query)
        
        data = {
            "inputs": query,
            "parameters": {"max_new_tokens": 1024},
            "multimodal_params": {"images": images}
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            print(response.json())
        else:
            print('Error:', response.status_code, response.text)

    finally:
        # Ensure the server process is terminated when done
        server_process.terminate()

if __name__ == "__main__":
    main()
