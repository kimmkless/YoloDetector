import os
import shutil
import json
import xml.etree.ElementTree as ET

def get_class_names_from_voc(voc_dir):
    class_set = set()
    for root_dir, _, files in os.walk(voc_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root_dir, file)
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for obj in root.findall("object"):
                        name = obj.find("name")
                        if name is not None:
                            class_set.add(name.text.strip())
                except Exception as e:
                    print(f"[XML ERROR] {xml_path}: {e}")
    return sorted(list(class_set))

def get_class_names_from_coco(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "categories" not in data:
        raise ValueError("COCO JSON 文件中未找到 'categories'")
    return [cat["name"] for cat in sorted(data["categories"], key=lambda x: x["id"])]

def convert_voc_to_yolo(voc_dir, save_root, class_names):
    save_dir = os.path.join(save_root, "train")
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    image_dir = os.path.join(voc_dir, "images")
    for root_dir, _, files in os.walk(voc_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root_dir, file)
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    filename = root.find("filename").text
                    width = int(root.find("size/width").text)
                    height = int(root.find("size/height").text)

                    txt_name = os.path.splitext(filename)[0] + ".txt"
                    txt_path = os.path.join(save_dir, "labels", txt_name)
                    img_path = os.path.normpath(os.path.join(image_dir, filename))

                    if not os.path.exists(img_path):
                        print(f"[SKIP] Image not found: {img_path}")
                        continue

                    shutil.copy(img_path, os.path.join(save_dir, "images", filename))

                    with open(txt_path, "w") as f:
                        for obj in root.findall("object"):
                            cls = obj.find("name").text
                            if cls not in class_names:
                                continue
                            cls_id = class_names.index(cls)
                            bbox = obj.find("bndbox")
                            xmin = int(bbox.find("xmin").text)
                            ymin = int(bbox.find("ymin").text)
                            xmax = int(bbox.find("xmax").text)
                            ymax = int(bbox.find("ymax").text)

                            x_center = (xmin + xmax) / 2 / width
                            y_center = (ymin + ymax) / 2 / height
                            box_w = (xmax - xmin) / width
                            box_h = (ymax - ymin) / height

                            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                except Exception as e:
                    print(f"[ERROR] Failed to convert {xml_path}: {e}")

def convert_coco_to_yolo(json_path, image_dir, save_root, class_names):
    save_dir = os.path.join(save_root, "train")
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id2filename = {img["id"]: img["file_name"] for img in data["images"]}
    id2size = {img["id"]: (img["width"], img["height"]) for img in data["images"]}
    anns = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        anns.setdefault(image_id, []).append(ann)

    for image_id, filename in id2filename.items():
        width, height = id2size[image_id]
        txt_path = os.path.join(save_dir, "labels", os.path.splitext(filename)[0] + ".txt")
        src_img = os.path.join(image_dir, filename)
        dst_img = os.path.join(save_dir, "images", filename)

        if not os.path.exists(src_img):
            print(f"[WARN] Image not found: {src_img}")
            continue

        shutil.copy(src_img, dst_img)

        with open(txt_path, "w") as f:
            for ann in anns.get(image_id, []):
                cat_id = ann["category_id"]
                cls_name = next((c["name"] for c in data["categories"] if c["id"] == cat_id), None)
                if cls_name not in class_names:
                    continue
                cls_id = class_names.index(cls_name)
                x, y, w, h = ann["bbox"]
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w / width:.6f} {h / height:.6f}\n")

def write_data_yaml(save_dir, class_names):
    yaml_path = os.path.join(save_dir, "data.yaml")
    path_str = os.path.abspath(save_dir).replace("\\", "/")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {path_str}\n")
        f.write("train: train/images\n")
        f.write("val: train/images\n")  # 若有 val 可改为 val/images
        f.write(f"nc: {len(class_names)}\n")
        f.write("names: [" + ", ".join(f"'{cls}'" for cls in class_names) + "]\n")

def convert_voc_dataset(voc_dir):
    print(f"[INFO] 正在处理 VOC 数据集：{voc_dir}")
    class_names = get_class_names_from_voc(voc_dir)
    print(f"[INFO] 检测到类别: {class_names}")

    yolo_dir = os.path.normpath(os.path.join(voc_dir, "yolo_format"))
    convert_voc_to_yolo(voc_dir, yolo_dir, class_names)
    write_data_yaml(yolo_dir, class_names)
    print(f"[DONE] VOC 转换完成，YOLO 格式保存在: {yolo_dir}")
    return yolo_dir.replace("\\", "/"), class_names

def convert_coco_dataset(coco_dir):
    print(f"[INFO] 正在处理 COCO 数据集：{coco_dir}")
    json_path = os.path.join(coco_dir, "annotations.json")
    image_dir = os.path.join(coco_dir, "images")
    yolo_dir = os.path.join(coco_dir, "yolo_format")

    if not os.path.exists(json_path):
        raise FileNotFoundError("COCO 数据集中缺少 annotations.json")

    class_names = get_class_names_from_coco(json_path)
    convert_coco_to_yolo(json_path, image_dir, yolo_dir, class_names)
    write_data_yaml(yolo_dir, class_names)
    print(f"[DONE] COCO 转换完成，YOLO 格式保存在: {yolo_dir}")
    return yolo_dir.replace("\\", "/"), class_names