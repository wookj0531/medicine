import cv2
import numpy as np
import os

def force_black_to_white(image_path, threshold=60):
    """
    이미지에서 특정 밝기(threshold) 이하인 어두운 픽셀을 
    전부 완전한 흰색(255, 255, 255)으로 바꿔버립니다.
    
    Args:
        image_path (str): 파일 경로
        threshold (int): 기준 밝기 (0~255). 
                         이 값보다 어두우면 배경으로 간주하고 지웁니다.
                         테두리가 남으면 이 값을 올리세요. (기본 60 추천)
    """
    # 1. 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 이미지를 찾을 수 없습니다: {image_path}")
        return

    # 2. 밝기 확인을 위해 그레이스케일 변환
    # (RGB 평균으로 따져도 되지만, 그레이스케일이 밝기 판단엔 가장 정확합니다)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 마스크 생성: 기준값(threshold)보다 어두운 픽셀 찾기
    # 예: 밝기가 60보다 작은(어두운) 모든 픽셀을 True로 설정
    mask = gray < threshold

    # 4. 해당 픽셀들을 흰색으로 덮어쓰기
    # img[mask]는 마스크가 True인 위치의 픽셀들입니다.
    img[mask] = [255, 255, 255]

    # 5. 파일 저장
    file_dir, file_name = os.path.split(image_path)
    name, ext = os.path.splitext(file_name)
    
    # 구분하기 쉽게 _clean_white 라고 이름 붙임
    output_filename = f"{name}_clean_white.jpg"
    output_path = os.path.join(file_dir, output_filename)
    
    cv2.imwrite(output_path, img)
    print(f"✅ 변환 완료! (Threshold={threshold})")
    print(f"📂 저장된 파일: {output_path}")
    
    return img

# ==========================================
# 실행 부분
# ==========================================

# 1. threshold=60 : 기준을 꽤 느슨하게 잡았습니다.
# (검은색 ~ 진한 회색까지 전부 흰색으로 바뀝니다.)
# 만약 여전히 테두리가 보이면 80, 90으로 올려보세요.
# 세포가 지워지기 직전까지 올리는 것이 가장 깨끗합니다.

input_file = "Cell-Counting\Contour_method\sample_2.jpg"
# input_file = "Cell-Counting/image_523724.jpg" 

cleaned_img = force_black_to_white(input_file, threshold=60)

# 결과 눈으로 확인
if cleaned_img is not None:
    cv2.imshow("Cleaned Result", cleaned_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()