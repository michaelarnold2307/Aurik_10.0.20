import os

import requests


def main():
    url = "http://127.0.0.1:8000/process/"
    token = os.environ.get("AURIK_ADMIN_TOKEN", "secret-token")
    headers = {"Authorization": f"Bearer {token}"}
    data = {"input_path": "test1.wav", "output_path": "output_test1.wav", "policy": None}
    response = requests.post(url, json=data, headers=headers)
    print("Status:", response.status_code)
    print("Response:", response.json())


if __name__ == "__main__":
    main()
