import sys
import os
from ultralytics import YOLO

def main():
    model_path = sys.argv[1]
    yaml_path = sys.argv[2]

    model = YOLO(model_path)
    results = model.val(data=yaml_path)

    print("[RESULT_SAVE_DIR]", results.save_dir)  # 标记路径传回主进程

if __name__ == "__main__":
    main()
