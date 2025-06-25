import csv

def read_loss_from_results_csv(csv_path):
    loss_list = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    box_loss = float(row.get('train/box_loss', '') or 0)
                    cls_loss = float(row.get('train/cls_loss', '') or 0)
                    obj_loss = float(row.get('train/obj_loss', '') or 0)  # 仅 YOLOv5/7 使用

                    total_loss = box_loss + cls_loss + obj_loss

                    if total_loss > 0:
                        loss_list.append(total_loss)
                except ValueError:
                    continue  # 忽略非数字行
    except FileNotFoundError:
        return []

    return loss_list
