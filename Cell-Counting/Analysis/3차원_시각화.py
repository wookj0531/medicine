import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===============================
# 0. 이미지 로드
# ===============================
img = cv2.imread("Cell-Counting\Analysis\Sample_1.jpg") 
assert img is not None, "이미지를 불러올 수 없습니다."

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ===============================
# 1. 픽셀 샘플링 (중요)
# ===============================
h, w, _ = img_rgb.shape
pixels_rgb = img_rgb.reshape(-1, 3)
pixels_hsv = img_hsv.reshape(-1, 3)

# 너무 많으면 느리므로 랜덤 샘플링
num_samples = 15000
if pixels_rgb.shape[0] > num_samples:
    idx = np.random.choice(pixels_rgb.shape[0], num_samples, replace=False)
    pixels_rgb = pixels_rgb[idx]
    pixels_hsv = pixels_hsv[idx]

# 색상은 RGB 기준으로 시각화
colors = pixels_rgb / 255.0

# ===============================
# 2. RGB 3D Scatter
# ===============================
fig = plt.figure(figsize=(14,6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(
    pixels_rgb[:,0],
    pixels_rgb[:,1],
    pixels_rgb[:,2],
    c=colors,
    s=2
)

ax1.set_title("RGB Pixel Distribution (3D)")
ax1.set_xlabel("R")
ax1.set_ylabel("G")
ax1.set_zlabel("B")

# ===============================
# 3. HSV 3D Scatter
# ===============================
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(
    pixels_hsv[:,0],
    pixels_hsv[:,1],
    pixels_hsv[:,2],
    c=colors,
    s=2
)

ax2.set_title("HSV Pixel Distribution (3D)")
ax2.set_xlabel("Hue")
ax2.set_ylabel("Saturation")
ax2.set_zlabel("Value")

plt.tight_layout()
plt.show()