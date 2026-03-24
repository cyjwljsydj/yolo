import torch
import numpy as np

def parse_yolo_output(output, S=7, B=2, C=20, img_width=448, img_height=448, conf_threshold=0.5):
    """
    从 YOLOv1 模型输出中解析边界框坐标和类别标签

    Args:
        output: 模型输出张量 (batch_size, S, S, B*5 + C)
        S: 网格大小 (7)
        B: 每个网格的bbox数量 (2)
        C: 类别数量 (20)
        img_width, img_height: 原始图像尺寸 (用于反归一化)
        conf_threshold: 置信度阈值

    Returns:
        boxes: list of [x1, y1, x2, y2, confidence, class_id]
    """
    batch_size = output.shape[0]
    boxes = []

    for b in range(batch_size):
        batch_boxes = []
        pred = output[b]  # (S, S, 30)

        for i in range(S):
            for j in range(S):
                # 解析两个bbox
                for bbox_idx in range(B):
                    # 置信度
                    conf_idx = 0 if bbox_idx == 0 else 5
                    confidence = torch.sigmoid(pred[i, j, conf_idx])

                    # bbox参数 (tx, ty, tw, th)
                    bbox_start = 1 + bbox_idx * 5
                    tx, ty, tw, th = pred[i, j, bbox_start:bbox_start+4]

                    # 反sigmoid tx, ty
                    bx = torch.sigmoid(tx)
                    by = torch.sigmoid(ty)

                    # 计算中心点 (相对坐标)
                    cx = (j + bx) / S
                    cy = (i + by) / S

                    # 计算宽高 (相对坐标)
                    pw = 1.0  # 预设anchor, 这里简化用1.0
                    ph = 1.0
                    bw = pw * torch.exp(tw)
                    bh = ph * torch.exp(th)

                    # 转换为绝对像素坐标
                    x_center = cx * img_width
                    y_center = cy * img_height
                    w = bw * img_width
                    h = bh * img_height

                    # 转换为x1,y1,x2,y2
                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x_center + w / 2
                    y2 = y_center + h / 2

                    # 类别概率
                    class_probs = pred[i, j, 10:30]  # 20类
                    class_prob, class_id = torch.max(torch.softmax(class_probs, dim=0), dim=0)

                    # 最终置信度 = bbox_conf * class_prob
                    final_conf = confidence * class_prob

                    if final_conf > conf_threshold:
                        batch_boxes.append([
                            x1.item(), y1.item(), x2.item(), y2.item(),
                            final_conf.item(), class_id.item()
                        ])

        boxes.append(batch_boxes)

    return boxes

# 使用示例
if __name__ == "__main__":
    from yolov1 import YOLO

    model = YOLO()
    model.eval()

    # 模拟输入
    x = torch.randn(1, 3, 448, 448)
    with torch.no_grad():
        output = model(x)

    # 解析输出
    parsed_boxes = parse_yolo_output(output, conf_threshold=0.3)

    print("检测到的框数量:", len(parsed_boxes[0]))
    for box in parsed_boxes[0][:5]:  # 显示前5个
        x1, y1, x2, y2, conf, cls = box
        print(".2f")