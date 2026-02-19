import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 0. 이미지 로드
# ===============================
img = cv2.imread("Cell-Counting\Contour_method\sample_2.jpg")
assert img is not None, "이미지를 불러올 수 없습니다."

orig = img.copy()

# ===============================
# 1. Grayscale
# ===============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===============================
# 2. k-means (배경 / 적혈구 분리)
# ===============================
Z = gray.reshape((-1, 1)).astype(np.float32)

K = 2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

_, labels, centers = cv2.kmeans(
    Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

segmented = centers[labels.flatten()].reshape(gray.shape)
segmented = segmented.astype(np.uint8)

# ===============================
# 3. Binary mask (적혈구 = 흰색)
# ===============================
_, binary = cv2.threshold(
    segmented, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# ===============================
# 3-1. Hole filling (적혈구 내부 구멍 제거)
# ===============================
h, w = binary.shape
mask = np.zeros((h + 2, w + 2), np.uint8)

binary_filled = binary.copy()
mask = np.zeros((h + 2, w + 2), np.uint8)

# 테두리 전체를 seed로 flood fill
for x in range(w):
    cv2.floodFill(binary_filled, mask, (x, 0), 255)
    cv2.floodFill(binary_filled, mask, (x, h-1), 255)

for y in range(h):
    cv2.floodFill(binary_filled, mask, (0, y), 255)
    cv2.floodFill(binary_filled, mask, (w-1, y), 255)

binary_filled_inv = cv2.bitwise_not(binary_filled)
binary = binary | binary_filled_inv

# ===============================
# 4. Morphology (노이즈 제거)
# ===============================
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=2)

# ===============================
# 4-1. Contour-based hole filling (강제 채우기)
# ===============================
binary_filled = np.zeros_like(opening)

cnts, _ = cv2.findContours(
    opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

for cnt in cnts:
    area = cv2.contourArea(cnt)
    if area < 30:   # 기존 면적 필터와 맞추기
        continue
    cv2.drawContours(binary_filled, [cnt], -1, 255, -1)

binary = binary_filled


# ===============================
# 5. Distance Transform
# ===============================
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# ===============================
# 6. Marker 생성
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
# 7. Watershed
# ===============================
markers = cv2.watershed(img, markers)

# ===============================
# 8. Contour 추출 & 감염 판별
# ===============================
INFECTED_MEAN_THRESHOLD = 180 # ← 튜닝용

output = orig.copy()
rbc_count = 0
infected_count = 0

for label in np.unique(markers):
    if label <= 1:
        continue

    mask = np.zeros(gray.shape, dtype="uint8")
    mask[markers == label] = 255

    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(cnts) == 0:
        continue

    cnt = max(cnts, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    if area < 30:
        continue

    # 평균 픽셀값
    mean_intensity = cv2.mean(gray, mask=mask)[0]

    # 감염 여부
    if mean_intensity <= INFECTED_MEAN_THRESHOLD:
        color = (255, 0, 0)   # 감염
        infected_count += 1
    else:
        color = (0, 255, 0)   # 정상

    cv2.drawContours(output, [cnt], -1, color, 1)
    rbc_count += 1

print("총 적혈구 수:", rbc_count)
print("감염된 적혈구 수:", infected_count)


# ===============================
# 9. 결과 시각화
# ===============================
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

ax[0,0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
ax[0,0].set_title("Original")

ax[0,1].imshow(gray, cmap="gray")
ax[0,1].set_title("Grayscale")

ax[0,2].imshow(binary, cmap="gray")
ax[0,2].set_title("Binary (Hole Filled)")

ax[1,0].imshow(binary, cmap="gray")
ax[1,0].set_title("Binary mask")

ax[1,1].imshow(dist, cmap="jet")
ax[1,1].set_title("Distance Transform")

ax[1,2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
ax[1,2].set_title(f"Count green-contour (Count = {rbc_count}) (Infected = {infected_count})")

for a in ax.flatten():
    a.axis("off")

plt.tight_layout()
plt.show()
