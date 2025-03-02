from PIL import Image
import numpy as np

# 加载图片文件
image_path = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/debug/semantic_t2639_f51.png"  # 替换为你的图片文件路径
image = Image.open(image_path)

# 将图片转换为 NumPy 数组
image_array = np.array(image)

# 如果是 RGB 图片，展平每个像素的颜色值
if len(image_array.shape) == 3:  # RGB 图像
    # 获取每个像素的 (R, G, B) 值作为元组
    unique_colors = np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0)
else:  # 单通道灰度图像
    unique_colors = np.unique(image_array)

# 打印颜色分布
print("Unique colors in the image:")
for color in unique_colors:
    print(color)