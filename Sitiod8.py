import csv
import json
from aifc import open
from builtins import print, int

import pymongo

def convert_log_to_csv(log_file, csv_file):
    with open(log_file, 'r') as file:
        log_data = file.readlines()

    csv_data = []
    for line in log_data:
        fields = line.strip().split(',')
        csv_data.append(fields)

    with open(csv_file, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(csv_data)

    print(f"Log file converted to CSV: {csv_file}")

def convert_log_to_json(log_file, json_file):
    with open(log_file, 'r') as file:
        log_data = file.readlines()

    json_data = []
    for line in log_data:
        fields = line.strip().split(',')
        json_data.append({
            'URL': fields[0],
            'IP': fields[1],
            'timeStamp': fields[2],
            'timeSpent': int(fields[3])
        })

    with open(json_file, 'w') as file:
        json.dump(json_data, file)

    print(f"Log file converted to JSON: {json_file}")

# Путь к исходному файлу лога
log_file = 'D:\\webserver_log.txt'

# Путь и имя файла CSV для сохранения
csv_file = 'webserver_log.csv'

# Путь и имя файла JSON для сохранения
json_file = 'webserver_log.json'

# Вызов функций для преобразования лога в CSV и JSON
convert_log_to_csv(log_file, csv_file)
convert_log_to_json(log_file, json_file)


def load_json_to_mongodb(json_file):
    # Подключение к MongoDB
    client = pymongo.MongoClient('mongodb://localhost:27017/')

    # Получение базы данных
    db = client['mydatabase']

    # Получение коллекции, в которую будут загружены данные
    collection = db['logs']

    # Открытие JSON-файла
    with open(json_file, 'r') as file:
        logs = json.load(file)

        # Вставка данных в MongoDB
        collection.insert_many(logs)

    print("Data loaded into MongoDB")

# Путь и имя файла JSON
json_file = 'webserver_log.json'

# Вызов функции для загрузки данных в MongoDB
load_json_to_mongodb(json_file)