from ultralytics import YOLO
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

# 加载模型
model = YOLO('../Models/PlantsDetector.pt')  # 加载自定义的训练模型

if __name__ == '__main__':
    # 对模型进行验证
    metrics = model.val()  # 调用val方法进行模型验证，不需要传入参数，数据集和设置已被模型记住

    # 输出不同的性能指标
    print("AP (mAP@0.5:0.95):", metrics.box.map)  # 输出平均精度均值（AP，Average Precision）在IoU阈值从0.5到0.95的范围内的结果
    print("AP@0.5 (mAP@0.5):", metrics.box.map50)  # 输出在IoU=0.5时的平均精度（AP50）
    print("AP@0.75 (mAP@0.75):", metrics.box.map75)  # 输出在IoU=0.75时的平均精度（AP75）
    print("APs per category (mAP@0.5:0.95 per category):", metrics.box.maps)  # 输出每个类别在IoU阈值从0.5到0.95的平均精度的列表





