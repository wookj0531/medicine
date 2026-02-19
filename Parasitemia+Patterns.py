import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 0. 이미지 로드
# ===============================
image_path = "sample_infected_3.jpg"
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

# ===============================
# 2. Masked Otsu Segmentation
# ===============================
# 설명: 전체 이미지가 아니라, '배경이 아닌 부분'만 가지고 Otsu를 계산합니다.

# 2-1. 1차 분류: 흰색 배경(255 근처)과 나머지(혈청+적혈구) 분리
ret_bg, mask_bg = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
mask_tissue = cv2.bitwise_not(mask_bg) # 배경 반전 -> 조직만 남김

# 2-2. 2차 분류: 조직 영역(혈청+적혈구) 내부에서만 Otsu 계산
tissue_pixels = gray_enhanced[mask_tissue > 0]

# 뽑아낸 픽셀들만 가지고 Otsu 임계값을 계산합니다.
otsu_threshold, _ = cv2.threshold(
    tissue_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print(f"Otsu 자동값: {otsu_threshold}")

_, binary = cv2.threshold(gray_enhanced, otsu_threshold, 255, cv2.THRESH_BINARY_INV)

# [중요] 배경이었던 곳도 흰색으로 변했을 수 있으니, 확실하게 날려버립니다.
binary = cv2.bitwise_and(binary, binary, mask=mask_tissue)


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

# ===============================
# 5. 결과 출력 (Infected Logic 개선 + 패턴 추출 추가)
# ===============================
output = orig.copy()
count = 0
infected_count = 0

# 파라미터 설정
ROI_RADIUS = 15  # 세포 크기에 맞춰 조정 필요 (반지름)
PARASITE_THR = 10 # 검출된 보라색 픽셀이 이 개수 이상이면 감염으로 간주

# Giemsa 염색된 기생충(보라색/진한 파란색)을 잡기 위한 HSV 범위
lower_purple = np.array([120, 100, 50])
upper_purple = np.array([170, 255, 255])

# 원본 이미지를 HSV로 변환 (색상 분석용)
hsv_img = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)

# [NEW] 감염된 세포의 패턴 이미지를 저장할 리스트
infected_rois_vis = []

for (y, x) in filtered_peaks:
    # 1. 세포 한 개에 대한 ROI(관심 영역) 설정
    y_min = max(0, y - ROI_RADIUS)
    y_max = min(orig.shape[0], y + ROI_RADIUS)
    x_min = max(0, x - ROI_RADIUS)
    x_max = min(orig.shape[1], x + ROI_RADIUS)
    
    cell_roi_hsv = hsv_img[y_min:y_max, x_min:x_max]
    
    # 2. ROI 내에서 원형 마스크 생성
    mask_roi = np.zeros((cell_roi_hsv.shape[0], cell_roi_hsv.shape[1]), dtype=np.uint8)
    cv2.circle(mask_roi, (x - x_min, y - y_min), ROI_RADIUS, 255, -1)
    
    # 3. 색상 기반 기생충 검출 (HSV Masking)
    parasite_mask = cv2.inRange(cell_roi_hsv, lower_purple, upper_purple)
    parasite_mask = cv2.bitwise_and(parasite_mask, parasite_mask, mask=mask_roi)
    
    # 4. 감염 여부 판단
    detected_pixels = cv2.countNonZero(parasite_mask)
    
    if detected_pixels > PARASITE_THR:
        color = (0, 0, 255) # Red for Infected
        infected_count += 1
        
        # [NEW] 감염된 경우, 시각화를 위해 Grayscale(CLAHE 적용본) ROI 추출
        # 패턴 분석에는 명암 대비가 뚜렷한 gray_enhanced가 효과적입니다.
        roi_vis = gray_enhanced[y_min:y_max, x_min:x_max]
        infected_rois_vis.append(roi_vis)
        
    else:
        color = (0, 255, 0) # Green for Healthy
    
    # 시각화
    cv2.circle(output, (x, y), ROI_RADIUS, color, 2)
    count += 1

# Parasitemia 계산
parasitemia = (infected_count / count * 100) if count > 0 else 0

print(f"Total Cells: {count}")
print(f"Infected Cells: {infected_count}")
print(f"Parasitemia: {parasitemia:.2f}%")

# 결과 시각화 1: 전체 카운팅 결과
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title(f"Result (Parasitemia: {parasitemia:.2f}%)")
plt.axis("off")
plt.show()

# ===============================
#  7-2. 감염된 적혈구 히트맵 시각화
# ===============================
if infected_rois_vis:
    # 너무 많으면 최대 6개까지만 시각화
    n_show = min(len(infected_rois_vis), 6)
    
    # subplot 생성
    fig, ax = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    
    # 감염된 세포가 1개인 경우 배열 차원 보정
    if n_show == 1: 
        ax = ax.reshape(2, 1)

    # --- 패턴 추출 파라미터 (조정 가능) ---
    # 밝은 부분은 배경/세포질로 보고 날리고, 아주 어두운(기생충) 부분만 남김
    # CLAHE가 적용되었으므로 값이 달라질 수 있어, 필요시 이 값을 조정하세요.
    DARK_CUT_OFF = 100  
    # ------------------------------------

    for i in range(n_show):
        roi_target = infected_rois_vis[i].copy()
        
        # 1. 감염된 세포 원본(Grayscale/CLAHE) 표시
        ax[0, i].imshow(roi_target, cmap="gray")
        ax[0, i].set_title(f"Infected #{i+1}")
        ax[0, i].axis("off")

        # 2. 어두운 특이점(기생충 패턴) 추출
        # 마스크 생성: DARK_CUT_OFF보다 어두운 부분만 True
        mask = roi_target < DARK_CUT_OFF
        
        # 배경을 흰색(255)으로 날려서 어두운 패턴만 남김
        roi_refined = np.full_like(roi_target, 255) 
        roi_refined[mask] = roi_target[mask] 

        # 3. 히트맵 시각화 (magma 컬러맵)
        # 어두울수록(값이 낮을수록) 진한 색상으로 표현됨
        ax[1, i].imshow(roi_refined, cmap="magma")
        ax[1, i].set_title(f"Infected Pattern")
        ax[1, i].axis("off")
        
    plt.suptitle("Infected Cell Patterns (Extracted from HSV Detection)", fontsize=16)
    plt.tight_layout()
    plt.show()
else:
    print("감염된 세포가 검출되지 않아 패턴 시각화를 건너뜁니다.")