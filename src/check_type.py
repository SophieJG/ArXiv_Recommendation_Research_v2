import json

def check_type(path: str):
    """
    Check the type of the data in the file.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for key, value in data.items():
        print(key, type(key))
        print(value, type(value))
        break


if __name__ == "__main__":
    check_type("Data/abstract/abstract.json")
