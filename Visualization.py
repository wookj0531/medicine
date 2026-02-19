import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 0. 이미지 로드
# ===============================
image_path = "Cell-Counting/sample_infected_2.jpg"
img = cv2.imread(image_path)
if img is None:
    raise ValueError("이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
orig = img.copy()

# ===============================
# 1. Grayscale & CLAHE
# ===============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 흐릿한 적혈구 대비 향상 (필수)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray_enhanced = clahe.apply(gray)

# [NEW] 랩미팅용 시각화 1: Preprocessing 단계
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)); axes[0].set_title("1. Original Image")
axes[1].imshow(gray, cmap='gray'); axes[1].set_title("2. Grayscale")
axes[2].imshow(gray_enhanced, cmap='gray'); axes[2].set_title("3. CLAHE Enhanced (Contrast+)")
for ax in axes: ax.axis("off")
plt.suptitle("Step 1: Image Preprocessing", fontsize=16)
plt.tight_layout()
plt.show()

# ===============================
# 2. Masked Otsu Segmentation
# ===============================
# 2-1. 1차 분류: 흰색 배경(255 근처)과 나머지(혈청+적혈구) 분리
ret_bg, mask_bg = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
mask_tissue = cv2.bitwise_not(mask_bg) # 배경 반전 -> 조직만 남김

# 2-2. 2차 분류: 조직 영역(혈청+적혈구) 내부에서만 Otsu 계산
tissue_pixels = gray_enhanced[mask_tissue > 0]
otsu_threshold, _ = cv2.threshold(
    tissue_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print(f"Otsu 자동값: {otsu_threshold}")

_, binary = cv2.threshold(gray_enhanced, otsu_threshold, 255, cv2.THRESH_BINARY_INV)
binary = cv2.bitwise_and(binary, binary, mask=mask_tissue)
binary_otsu = binary.copy() # 시각화를 위해 백업

# [NEW] 랩미팅용 시각화 2: Segmentation 단계
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(mask_bg, cmap='gray'); axes[0].set_title("1. Background Mask")
axes[1].imshow(mask_tissue, cmap='gray'); axes[1].set_title("2. Tissue Mask (Bg Removed)")
axes[2].imshow(binary_otsu, cmap='gray'); axes[2].set_title("3. Masked Otsu Binary")
for ax in axes: ax.axis("off")
plt.suptitle("Step 2: Masked Otsu Segmentation", fontsize=16)
plt.tight_layout()
plt.show()

# ===============================
# 3. 노이즈 제거 및 구멍 채우기
# ===============================
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

cnts, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
binary_final = np.zeros_like(opening)
total_area = opening.shape[0] * opening.shape[1]

for cnt in cnts:
    area = cv2.contourArea(cnt)
    if area < 30: continue # 먼지 제거
    if area > (total_area * 0.20): continue # 큰 테두리 제거
    cv2.drawContours(binary_final, [cnt], -1, 255, -1)

binary = binary_final

# [NEW] 랩미팅용 시각화 3: 노이즈 제거 단계
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(binary_otsu, cmap='gray'); axes[0].set_title("1. Initial Binary")
axes[1].imshow(opening, cmap='gray'); axes[1].set_title("2. Morphology Opening (Dust removal)")
axes[2].imshow(binary_final, cmap='gray'); axes[2].set_title("3. Final Clean Mask (Contour filtered)")
for ax in axes: ax.axis("off")
plt.suptitle("Step 3: Noise Removal & Filtering", fontsize=16)
plt.tight_layout()
plt.show()

# ===============================
# 4. Distance Transform & Counting
# ===============================
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
dist = cv2.GaussianBlur(dist, (7, 7), 0)

min_peak_ratio = 0.40
local_max = cv2.dilate(dist, kernel) == dist
local_max &= (dist > min_peak_ratio * dist.max())

num_peaks, peak_labels = cv2.connectedComponents(local_max.astype(np.uint8))
num_peaks -= 1

peak_points = []
for i in range(1, num_peaks + 1):
    ys, xs = np.where(peak_labels == i)
    if len(xs) == 0: continue
    peak_points.append((int(np.mean(ys)), int(np.mean(xs))))

# NMS (겹침 제거)
min_peak_distance = 20
peak_points = sorted(peak_points, key=lambda p: dist[p[0], p[1]], reverse=True)
filtered_peaks = []
for (y, x) in peak_points:
    keep = True
    for (fy, fx) in filtered_peaks:
        if np.hypot(y - fy, x - fx) < min_peak_distance: keep = False; break
    if keep: filtered_peaks.append((y, x))

# [NEW] 랩미팅용 시각화 4: 겹친 세포 분리 과정 (매우 중요)
peak_vis = np.zeros_like(gray)
for (y, x) in filtered_peaks:
    cv2.circle(peak_vis, (x, y), 3, 255, -1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(binary, cmap='gray'); axes[0].set_title("1. Clean Binary Mask")
axes[1].imshow(dist, cmap='jet'); axes[1].set_title("2. Distance Transform (Heatmap)")
axes[2].imshow(peak_vis, cmap='gray'); axes[2].set_title("3. Detected Cell Centers (Peaks)")
for ax in axes: ax.axis("off")
plt.suptitle("Step 4: Overlapping Cells Separation (Distance Transform)", fontsize=16)
plt.tight_layout()
plt.show()

# ===============================
# 5. 결과 출력 (Infected Logic 개선 + 패턴 추출 추가)
# ===============================
output = orig.copy()
count = 0
infected_count = 0

ROI_RADIUS = 15  
PARASITE_THR = 10 
lower_purple = np.array([120, 100, 50])
upper_purple = np.array([170, 255, 255])
hsv_img = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
infected_rois_vis = []

for (y, x) in filtered_peaks:
    y_min = max(0, y - ROI_RADIUS)
    y_max = min(orig.shape[0], y + ROI_RADIUS)
    x_min = max(0, x - ROI_RADIUS)
    x_max = min(orig.shape[1], x + ROI_RADIUS)
    
    cell_roi_hsv = hsv_img[y_min:y_max, x_min:x_max]
    mask_roi = np.zeros((cell_roi_hsv.shape[0], cell_roi_hsv.shape[1]), dtype=np.uint8)
    cv2.circle(mask_roi, (x - x_min, y - y_min), ROI_RADIUS, 255, -1)
    
    parasite_mask = cv2.inRange(cell_roi_hsv, lower_purple, upper_purple)
    parasite_mask = cv2.bitwise_and(parasite_mask, parasite_mask, mask=mask_roi)
    detected_pixels = cv2.countNonZero(parasite_mask)
    
    if detected_pixels > PARASITE_THR:
        color = (0, 0, 255) # Red for Infected
        infected_count += 1
        roi_vis = gray_enhanced[y_min:y_max, x_min:x_max]
        infected_rois_vis.append(roi_vis)
    else:
        color = (0, 255, 0) # Green for Healthy
    
    cv2.circle(output, (x, y), ROI_RADIUS, color, 2)
    count += 1

parasitemia = (infected_count / count * 100) if count > 0 else 0

# ===============================
# 6. 최종 결과 및 패턴 시각화 (기존 유지 + 제목 추가)
# ===============================
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title(f"Step 5: Final Result (Total: {count} | Infected: {infected_count} | Parasitemia: {parasitemia:.2f}%)", fontsize=14)
plt.axis("off")
plt.show()

if infected_rois_vis:
    n_show = min(len(infected_rois_vis), 6)
    fig, ax = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    if n_show == 1: ax = ax.reshape(2, 1)

    DARK_CUT_OFF = 100  

    for i in range(n_show):
        roi_target = infected_rois_vis[i].copy()
        
        ax[0, i].imshow(roi_target, cmap="gray")
        ax[0, i].set_title(f"Infected #{i+1}")
        ax[0, i].axis("off")

        mask = roi_target < DARK_CUT_OFF
        roi_refined = np.full_like(roi_target, 255) 
        roi_refined[mask] = roi_target[mask] 

        ax[1, i].imshow(roi_refined, cmap="magma")
        ax[1, i].set_title(f"Parasite Pattern")
        ax[1, i].axis("off")
        
    plt.suptitle("Step 6: Cropped Infected Cell Patterns (Bridge to Objective 2)", fontsize=16)
    plt.tight_layout()
    plt.show()