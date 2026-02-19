# 📌 Project Title
> Blood Cell Counting Method
<br>

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

---

## 📖 Overview
Cell Counting 관련 코드 정리 

- 🔍 Problem: 적혈구 세기 
- 💡 Solution: Analysis, DT_peak_method, Countour_method

---

## ⭐ Key Features
- ✅ Analysis: 이미지의 색상 분포 분석용
<img width="4210" height="1209" alt="그림2" src="https://github.com/user-attachments/assets/553f1829-7957-4f49-96d8-fa98d183638e" /><br><br>

  
- ✅ DT_peak_method
<img width="4244" height="1355" alt="그림1" src="https://github.com/user-attachments/assets/b7f21a8d-6fc9-4626-88f8-a7c443b305a0" /><br><br>

  
- ✅ Countour_method
<img width="4196" height="1368" alt="그림3" src="https://github.com/user-attachments/assets/ceba599d-11a3-4096-8fc5-b6e550247c72" /><br><br>

---

## 🏗 Project Structure
```bash
Project/
├── Analysis/              
│   ├── 3차원_시각화.py
│   └── 히스토그램.py           
├── Contour_method/               
│   ├── 개수세기_통합(kmeans-watershed).py
│   └── 개수세기_통합(kmeans-watershed)_hmap.py
├── DT_peak_method/
│   ├── 개수세기_circle_감염_hmap.py
│   └── 개수세기_통합(blue-circle)_GY_hyper.py    
├── Dummy/ # 필요없는 파일 모음 (무시하기)    
├── requirements.txt
├──.gitignore
└── README.md
```

---

## ⚙️ Installation
```bash
git clone https://github.com/lko9911/Cell-Counting.git
cd Cell-Counting
pip install -r requirements.txt
```
