import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 0. 이미지 로드 + 원형 영역 Crop
# ===============================
img_full = cv2.imread("Cell-Counting\Contour_method\sample_2.jpg")
assert img_full is not None, "이미지를 불러올 수 없습니다."

H, W = img_full.shape[:2]
center_full = (W // 2, H // 2)
radius_full = min(center_full[0], center_full[1])

# 정사각형 crop
x0 = center_full[0] - radius_full
y0 = center_full[1] - radius_full
x1 = center_full[0] + radius_full
y1 = center_full[1] + radius_full

img = img_full[y0:y1, x0:x1].copy()
orig = img.copy()

h, w = img.shape[:2]
center = (w // 2, h // 2)
radius = min(center)

# 원형 mask
Y, X = np.ogrid[:h, :w]
circle_mask = (X - center[0])**2 + (Y - center[1])**2 <= radius**2

# ===============================
# 1. Grayscale
# ===============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===============================
# 2. k-means (원 내부만 사용)
# ===============================
Z = gray[circle_mask].reshape((-1, 1)).astype(np.float32)

K = 2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)

_, labels, centers = cv2.kmeans(
    Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

centers = centers.flatten()

# 2D 매핑
segmented = np.zeros_like(gray)
segmented[circle_mask] = centers[labels.flatten()].astype(np.uint8)

# ===============================
# 3. Binary mask 생성
# ===============================
cell_cluster = np.argmin(centers)

binary = np.zeros_like(gray)
cell_pixels = (labels.flatten() == cell_cluster).astype(np.uint8) * 255
binary[circle_mask] = cell_pixels

# 원 바깥 0으로 설정
binary[~circle_mask] = 0

# ===============================
# 4. Hole Filling
# ===============================
mask_ff = np.zeros((h + 2, w + 2), np.uint8)
binary_filled = binary.copy()

for x in range(w):
    cv2.floodFill(binary_filled, mask_ff, (x, 0), 255)
    cv2.floodFill(binary_filled, mask_ff, (x, h - 1), 255)

for y in range(h):
    cv2.floodFill(binary_filled, mask_ff, (0, y), 255)
    cv2.floodFill(binary_filled, mask_ff, (w - 1, y), 255)

binary_filled_inv = cv2.bitwise_not(binary_filled)
binary = binary | binary_filled_inv

# ===============================
# 5. Morphology
# ===============================
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# 작은 잡음 제거 + 내부 채우기
binary_clean = np.zeros_like(binary)

cnts, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

for cnt in cnts:
    if cv2.contourArea(cnt) < 30:
        continue
    cv2.drawContours(binary_clean, [cnt], -1, 255, -1)

binary = binary_clean

# ===============================
# 6. Distance Transform
# ===============================
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

# ===============================
# 7. DT Peak Detection
# ===============================
min_peak_ratio = 0.20
kernel_peak = np.ones((3, 3), np.uint8)

local_max = cv2.dilate(dist, kernel_peak) == dist
local_max &= (dist > min_peak_ratio * dist.max())

num_peaks, peak_labels = cv2.connectedComponents(
    local_max.astype(np.uint8)
)
num_peaks -= 1

# ===============================
# 8. Peak centroid 계산
# ===============================
peak_points = []

for i in range(1, num_peaks + 1):
    ys, xs = np.where(peak_labels == i)
    if len(xs) == 0:
        continue

    cy = int(np.mean(ys))
    cx = int(np.mean(xs))
    peak_points.append((cy, cx))

# ===============================
# 9. Peak NMS
# ===============================
min_peak_distance = 13

peak_points = sorted(
    peak_points,
    key=lambda p: dist[p[0], p[1]],
    reverse=True
)

filtered_peaks = []

for (y, x) in peak_points:
    keep = True
    for (fy, fx) in filtered_peaks:
        if np.hypot(y - fy, x - fx) < min_peak_distance:
            keep = False
            break
    if keep:
        filtered_peaks.append((y, x))

# ===============================
# 10. 감염 판별
# ===============================
ROI_RADIUS = 15
INFECTED_MEAN_THRESHOLD = 170

output = orig.copy()

rbc_count = 0
infected_count = 0
mean_values = []

for (y, x) in filtered_peaks:

    roi_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(roi_mask, (x, y), ROI_RADIUS, 255, -1)

    mean_intensity = cv2.mean(gray, mask=roi_mask)[0]
    mean_values.append(mean_intensity)

    if mean_intensity <= INFECTED_MEAN_THRESHOLD:
        color = (0, 0, 255)   # 감염 (Red)
        infected_count += 1
    else:
        color = (0, 255, 0)   # 정상 (Green)

    cv2.circle(output, (x, y), ROI_RADIUS, color, 3)
    rbc_count += 1

print("적혈구 개수:", rbc_count)
print("감염된 적혈구 개수:", infected_count)

# ===============================
# 11. Visualization
# ===============================
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

ax[0,0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
ax[0,0].set_title("Original (Cropped)")

ax[0,1].imshow(segmented, cmap="gray")
ax[0,1].set_title("k-means result")

ax[0,2].imshow(binary, cmap="gray")
ax[0,2].set_title("Binary mask")

ax[1,0].imshow(dist, cmap="jet")
ax[1,0].set_title("Distance Transform")

ax[1,1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
ax[1,1].set_title(f"RBC={rbc_count}  Infected={infected_count}")


for a in ax.flatten():
    a.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.hist(mean_values, bins=30)
plt.xlabel("Mean Grayscale Intensity")
plt.ylabel("Count")
plt.title("RBC Intensity Distribution")
plt.show()
