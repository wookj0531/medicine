import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 0. 이미지 로드
# ===============================
# 배경이 하얀색으로 처리된 이미지를 불러옵니다.
image_path = "Cell-Counting\sample_infected_1.jpg" 
img = cv2.imread(image_path)
orig = img.copy()

# ===============================
# 1. Grayscale & CLAHE
# ===============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 흐릿한 적혈구 대비 향상 (필수)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray_enhanced = clahe.apply(gray)

# ===============================
# 2. Masked Otsu Segmentation (핵심!)
# ===============================
# 설명: 전체 이미지가 아니라, '배경이 아닌 부분'만 가지고 Otsu를 계산합니다.

# 2-1. 1차 분류: 흰색 배경(255 근처)과 나머지(혈청+적혈구) 분리
# 배경을 하얀색으로 만들었으므로 250 이상은 배경이라고 확신할 수 있습니다.
# mask_tissue: 혈청과 적혈구가 있는 영역 (True)
ret_bg, mask_bg = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
mask_tissue = cv2.bitwise_not(mask_bg) # 배경 반전 -> 조직만 남김

# 2-2. 2차 분류: 조직 영역(혈청+적혈구) 내부에서만 Otsu 계산
# mask_tissue가 0이 아닌(데이터가 있는) 픽셀값만 뽑아옵니다.
tissue_pixels = gray_enhanced[mask_tissue > 0]

# 뽑아낸 픽셀들만 가지고 Otsu 임계값을 계산합니다.
# 이렇게 하면 흰색 배경(255)의 간섭 없이 정확히 혈청과 적혈구를 나누는 값을 찾습니다.
otsu_threshold, _ = cv2.threshold(
    tissue_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


print(f"Otsu 자동값: {otsu_threshold}")

# 2. 수정한 값(loose_threshold)을 적용
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
# 5. 결과 출력 (Infected Logic 개선)
# ===============================
output = orig.copy()
count = 0
infected_count = 0

# 파라미터 설정
ROI_RADIUS = 15  # 세포 크기에 맞춰 조정 필요 (반지름)
PARASITE_THR = 10 # 검출된 보라색 픽셀이 이 개수 이상이면 감염으로 간주

# Giemsa 염색된 기생충(보라색/진한 파란색)을 잡기 위한 HSV 범위
# 보라색 계열: Hue가 대략 120~170 사이 (OpenCV 기준)
# 채도(S)와 명도(V)는 적절히 조정해야 합니다.
lower_purple = np.array([120, 100, 50])
upper_purple = np.array([170, 255, 255])

# 원본 이미지를 HSV로 변환 (색상 분석용)
hsv_img = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)

for (y, x) in filtered_peaks:
    # 1. 세포 한 개에 대한 ROI(관심 영역) 설정
    # 이미지 경계를 벗어나지 않도록 클리핑 처리
    y_min = max(0, y - ROI_RADIUS)
    y_max = min(orig.shape[0], y + ROI_RADIUS)
    x_min = max(0, x - ROI_RADIUS)
    x_max = min(orig.shape[1], x + ROI_RADIUS)
    
    cell_roi_hsv = hsv_img[y_min:y_max, x_min:x_max]
    
    # 2. ROI 내에서 원형 마스크 생성 (사각형 ROI 내의 원형 세포만 보기 위함)
    mask_roi = np.zeros((cell_roi_hsv.shape[0], cell_roi_hsv.shape[1]), dtype=np.uint8)
    cv2.circle(mask_roi, (x - x_min, y - y_min), ROI_RADIUS, 255, -1)
    
    # 3. 색상 기반 기생충 검출 (HSV Masking)
    # ROI 내에서 '보라색' 범위를 가진 픽셀만 추출
    parasite_mask = cv2.inRange(cell_roi_hsv, lower_purple, upper_purple)
    
    # 원형 마스크 바깥(사각형 모서리)의 노이즈 제거
    parasite_mask = cv2.bitwise_and(parasite_mask, parasite_mask, mask=mask_roi)
    
    # 4. 감염 여부 판단
    # 보라색 픽셀(기생충 추정)의 개수가 임계값(PARASITE_THR)보다 많으면 감염
    detected_pixels = cv2.countNonZero(parasite_mask)
    
    if detected_pixels > PARASITE_THR:
        color = (0, 0, 255) # Red for Infected
        infected_count += 1
        # 디버깅용: 감염된 곳에 점 찍어보기
        # output[y_min:y_max, x_min:x_max][parasite_mask > 0] = [0, 255, 255] 
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

# 결과 시각화
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title(f"Result (Parasitemia: {parasitemia:.2f}%)")
plt.axis("off")
plt.show()