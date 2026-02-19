import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 0. 이미지 로드
# ===============================
img = cv2.imread("Cell-Counting\Analysis\Sample_1.jpg")
assert img is not None, "이미지를 불러올 수 없습니다."

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ===============================
# 1. RGB 히스토그램
# ===============================
colors = ['r', 'g', 'b']
labels = ['Red', 'Green', 'Blue']

plt.figure(figsize=(12,4))
for i, (c, l) in enumerate(zip(colors, labels)):
    hist = cv2.calcHist([img_rgb], [i], None, [256], [0,256])
    plt.plot(hist, color=c, label=l)

plt.title("RGB Pixel Value Distribution")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# ===============================
# 2. HSV 히스토그램
# ===============================
hsv_labels = ['Hue', 'Saturation', 'Value']
hsv_colors = ['m', 'c', 'k']
ranges = [(0,180), (0,256), (0,256)]

plt.figure(figsize=(12,4))
for i, (l, col, r) in enumerate(zip(hsv_labels, hsv_colors, ranges)):
    hist = cv2.calcHist([img_hsv], [i], None, [256], r)
    plt.plot(hist, color=col, label=l)

plt.title("HSV Pixel Value Distribution")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()