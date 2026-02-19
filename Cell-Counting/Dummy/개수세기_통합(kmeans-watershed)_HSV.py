import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===============================
# 0. 이미지 로드
# ===============================
img = cv2.imread("Contour_method\Sample_1.png")
assert img is not None, "이미지를 불러올 수 없습니다."

orig = img.copy()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ===============================
# 1. HSV 전체 기반 k-means
# ===============================
Z = img_hsv.reshape((-1, 3)).astype(np.float32)

K = 2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

_, labels, centers = cv2.kmeans(
    Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

labels_2d = labels.reshape(img.shape[:2])

# ===============================
# 2. 적혈구 클러스터 선택 (HSV 전체 기준)
#    보통: Value가 낮거나 Saturation이 높은 쪽
# ===============================
# 여기서는 Value가 낮은 클러스터를 적혈구로 가정
rbc_cluster = np.argmin(centers[:, 2])

binary = np.zeros_like(labels_2d, dtype=np.uint8)
binary[labels_2d == rbc_cluster] = 255

# ===============================
# 3. Morphology
# ===============================
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=2)

# ===============================
# 4. Distance Transform
# ===============================
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# ===============================
# 5. Marker 생성
# ===============================
_, sure_fg = cv2.threshold(
    dist, 0.4 * dist.max(), 255, 0
)
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)

num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# ===============================
# 6. Watershed
# ===============================
markers = cv2.watershed(img, markers)

# ===============================
# 7. Contour & 개수 세기
# ===============================
output = orig.copy()
rbc_count = 0

for label in np.unique(markers):
    if label <= 1:
        continue

    mask = np.zeros(binary.shape, dtype="uint8")
    mask[markers == label] = 255

    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not cnts:
        continue

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area < 30:
        continue

    cv2.drawContours(output, [cnt], -1, (0, 255, 0), 1)
    rbc_count += 1

print("적혈구 개수:", rbc_count)

# ===============================
# 8. 결과 시각화
# ===============================
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

ax[0,0].imshow(img_rgb)
ax[0,0].set_title("Original")

ax[0,1].imshow(labels_2d, cmap="tab10")
ax[0,1].set_title("HSV k-means Labels")

ax[0,2].imshow(binary, cmap="gray")
ax[0,2].set_title("HSV Cluster Mask")

ax[1,0].imshow(dist, cmap="jet")
ax[1,0].set_title("Distance Transform")

ax[1,1].imshow(markers, cmap="tab20")
ax[1,1].set_title("Watershed Labels")

ax[1,2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
ax[1,2].set_title(f"Result (Count = {rbc_count})")

for a in ax.flatten():
    a.axis("off")

plt.tight_layout()
plt.show()

# ===============================
# 9. HSV 픽셀 3D 분포 시각화
# ===============================
pixels_hsv = img_hsv.reshape(-1, 3)
pixels_rgb = img_rgb.reshape(-1, 3)

num_samples = 15000
if pixels_hsv.shape[0] > num_samples:
    idx = np.random.choice(pixels_hsv.shape[0], num_samples, replace=False)
    pixels_hsv = pixels_hsv[idx]
    pixels_rgb = pixels_rgb[idx]

colors = pixels_rgb / 255.0

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    pixels_hsv[:,0],
    pixels_hsv[:,1],
    pixels_hsv[:,2],
    c=colors,
    s=2
)

ax.set_title("HSV Pixel Distribution (3D)")
ax.set_xlabel("Hue")
ax.set_ylabel("Saturation")
ax.set_zlabel("Value")

plt.tight_layout()
plt.show()
