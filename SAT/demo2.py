import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2

# 生成一个示例图像
image_size = 1000
image = np.ones((image_size, image_size))

# 在图像中心创建一个高亮区域
center_x, center_y = image_size // 2, image_size // 2
image[center_x - 100:center_x + 100, center_y - 100:center_y + 100] = 10

# 对图像进行高斯平滑处理
sigma = 100
smoothed_image = gaussian_filter(image, sigma=sigma)

# 绘制原始图像
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='viridis')
plt.colorbar()

# 绘制经过高斯平滑处理后的图像
plt.subplot(1, 2, 2)
plt.title('Smoothed Image (Gaussian Filter)')
plt.imshow(smoothed_image, cmap='viridis')
plt.colorbar()

plt.show()
