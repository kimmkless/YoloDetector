import csv

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
