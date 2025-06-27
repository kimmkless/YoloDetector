import os
import time
import csv

CURRENT_TIME = time.time()

def read_loss_from_results_csv(csv_path):
    loss_list = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    box_loss = float(row.get('train/box_loss', '') or 0)
                    cls_loss = float(row.get('train/cls_loss', '') or 0)
                    obj_loss = float(row.get('train/obj_loss', '') or 0)

                    total_loss = box_loss + cls_loss + obj_loss

                    if total_loss > 0:
                        loss_list.append(total_loss)
                except ValueError:
                    continue  # 忽略非数字行
    except FileNotFoundError:
        return []

    return loss_list

def get_latest_results_csv(base_dir='runs/train'):
    latest_csv = None
    latest_time = 0
    for root, dirs, files in os.walk(base_dir):
        if 'results.csv' in files:
            path = os.path.join(root, 'results.csv')
            mtime = os.path.getmtime(path)
            if mtime > latest_time and mtime > CURRENT_TIME:
                latest_time = mtime
                latest_csv = path
    return latest_csv

def get_latest_model_dir(base_dir='runs/train'):
    latest_dir = None
    latest_time = 0
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            full_path = os.path.join(root, d)
            if os.path.isdir(full_path):
                mtime = os.path.getmtime(full_path)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_dir = full_path
    return latest_dir

def read_metrics_from_results_csv(csv_path, extended=False):
    loss_list = []
    map50_list = []
    map5095_list = []
    precision_list = []
    recall_list = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    loss = float(row.get('loss', 0))
                    map50 = float(row.get('map50', 0))
                    map95 = float(row.get('map50-95', 0))
                    loss_list.append(loss)
                    map50_list.append(map50)
                    map5095_list.append(map95)

                    if extended:
                        precision = float(row.get('precision', 0)) or float(row.get('metrics/precision(B)', 0))
                        recall = float(row.get('recall', 0)) or float(row.get('metrics/recall(B)', 0))
                        precision_list.append(precision)
                        recall_list.append(recall)

                except (ValueError, TypeError):
                    continue
    except Exception as e:
        print("读取 CSV 出错：", e)

    if extended:
        return loss_list, map50_list, map5095_list, precision_list, recall_list
    else:
        return loss_list, map50_list, map5095_list
