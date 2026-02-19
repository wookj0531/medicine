import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 0. 이미지 로드
# ===============================
img = cv2.imread("Cell-Counting\sample_infected_1.jpg")
assert img is not None, "이미지를 불러올 수 없습니다."
orig = img.copy()

# ===============================
# 1. Grayscale & Preprocessing
# ===============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===============================
# 2. k-means
# ===============================
Z = gray.reshape((-1, 1)).astype(np.float32)
K = 2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

_, labels, centers = cv2.kmeans(
    Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)
segmented = centers[labels.flatten()].reshape(gray.shape).astype(np.uint8)

# ===============================
# 3. Binary mask
# ===============================
#_, binary = cv2.threshold(
#    segmented, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#)

cell_cluster = np.argmin(centers)  # 더 어두운 클러스터
binary = (labels.reshape(gray.shape) == cell_cluster).astype(np.uint8) * 255


# ===============================
# 4. Morphology
# ===============================
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

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
# 5. Distance Transform & Peak Detection
# ===============================
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

min_peak_ratio = 0.20
local_max = (cv2.dilate(dist, np.ones((3, 3))) == dist) & (dist > min_peak_ratio * dist.max())
num_peaks, peak_labels = cv2.connectedComponents(local_max.astype(np.uint8))

peak_points = []
for i in range(1, num_peaks):
    ys, xs = np.where(peak_labels == i)
    peak_points.append((int(np.mean(ys)), int(np.mean(xs))))

# NMS (중복 제거)
min_peak_distance = 13
peak_points = sorted(peak_points, key=lambda p: dist[p[0], p[1]], reverse=True)
filtered_peaks = []
for (y, x) in peak_points:
    if all(np.hypot(y - fy, x - fx) >= min_peak_distance for (fy, fx) in filtered_peaks):
        filtered_peaks.append((y, x))

# ===============================
# 6. 감염 분석 및 히트맵 추출
# ===============================
ROI_RADIUS = 7  # 히트맵을 보기 위해 반지름을 조금 키움
INFECTED_MEAN_THRESHOLD = 180  # 이 값보다 낮으면 감염(어두운 점 존재)

output = orig.copy()
infected_rois = []  # 감염된 세포의 이미지를 담을 리스트
rbc_count = 0
infected_count = 0

for (y, x) in filtered_peaks:
    # ROI 좌표 계산 (이미지 경계 확인)
    y1, y2 = max(0, y-ROI_RADIUS), min(gray.shape[0], y+ROI_RADIUS+1)
    x1, x2 = max(0, x-ROI_RADIUS), min(gray.shape[1], x+ROI_RADIUS+1)
    
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0: continue

    # 감염 판별용 마스크 (원의 내부만 계산)
    h_r, w_r = roi.shape
    roi_mask = np.zeros((h_r, w_r), dtype=np.uint8)
    cv2.circle(roi_mask, (x - x1, y - y1), ROI_RADIUS-2, 255, -1)
    
    mean_intensity = cv2.mean(roi, mask=roi_mask)[0]

    # 감염 여부에 따른 색상 결정 (빨간색: 감염, 초록색: 정상)
    if mean_intensity <= INFECTED_MEAN_THRESHOLD:
        color = (0, 0, 255) # Red
        infected_count += 1
        # 히트맵용 ROI 저장 (정규화하여 대비 극대화)
        roi_norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
        infected_rois.append(roi_norm)
    else:
        color = (0, 255, 0) # Green
    
    cv2.circle(output, (x, y), ROI_RADIUS, color, 2)
    rbc_count += 1

# ===============================
# 7. 최종 시각화
# ===============================

# [1] 전체 분석 결과
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
ax[0].set_title(f"Detection: Total={rbc_count}, Infected={infected_count}")
ax[1].imshow(dist, cmap="jet")
ax[1].set_title("Distance Transform (Peaks)")
for a in ax: a.axis("off")
plt.tight_layout()
plt.show()

# ===============================
# 7-2. 감염된 적혈구 히트맵 (원본 어두운 특이점 강조)
# ===============================
if infected_rois:
    n_show = min(len(infected_rois), 6)
    fig, ax = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    if n_show == 1: ax = ax.reshape(2, 1)

    # --- 핵심 설정 ---
    DARK_CUT_OFF = 80 
    # ----------------

    for i in range(n_show):
        # 1. 원본 그레이스케일 표시
        ax[0, i].imshow(infected_rois[i], cmap="gray")
        ax[0, i].set_title(f"Infected #{i+1}")

        # 2. 어두운 특이점 추출 (역전 없음)
        roi_target = infected_rois[i].copy()
        
        # 마스크 생성: DARK_CUT_OFF보다 밝은 부분은 제외
        mask = roi_target < DARK_CUT_OFF
        
        # 배경은 매우 밝게(255) 처리하거나, 혹은 0으로 두고 관심 영역만 유지
        # 여기서는 '어두운 점'을 강조하기 위해 나머지 영역을 255(흰색)로 채웁니다.
        roi_refined = np.full_like(roi_target, 255) 
        roi_refined[mask] = roi_target[mask] 

        # 3. 히트맵 시각화
        # 역전(255-)을 하지 않았으므로, 가장 어두운 점이 히트맵에서도 가장 낮은 값을 가집니다.
        im = ax[1, i].imshow(roi_refined, cmap="magma")
        ax[1, i].set_title(f"Infected Cell Pattern")
        
    for a in ax.flatten():
        a.axis("off")
        
    plt.suptitle("Infected pattern", fontsize=16)
    plt.tight_layout()
    plt.show()