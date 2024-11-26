# AI 기반 고객친화적 자동 맞춤 아이스크림 판매 서비스

AI가 자동으로 고객을 인식하고 혜택을 적용하며 아이스크림을 판매하는 서비스를 제작하였습니다.  
이 프로젝트는 고객 맞춤형 아이스크림 판매를 목표로 하며, 고객 인식 및 서비스 자동화를 위해 AI 기술을 활용합니다.
이 서비스는 **GUI**, **DB**, **CV**, **Robot arm**의 4개 파트로 구성되며, 각 파트는 **socket 통신**을 통해 연결되어 동작합니다.

---

## 주요 구성 요소
1. **GUI**: 사용자 인터페이스 제공
2. **DB**: 데이터베이스 관리
3. **CV**: 컴퓨터 비전 기반 고객 인식
4. **Robot arm**: 아이스크림 제공 (보안 상 코드 미포함)

---
## 프로젝트 팀 구성
| 이름       | 역할                                             |
|------------|--------------------------------------------------|
| 박현지     | 팀장 및 DB개발 담당, 총괄 및 프로젝트 일정관리  |
| 배장훈     | Robot arm 동작 개발 담당, 통신 및 코드 통합      |
| 김우식     | AI server(Computer vision 기반 기능) 개발       |
| 장우석     | GUI(Kiosk) 개발 담당                            |
| 윤정훈     | Chatbot 개발 담당                               |


---

## 실행 방법
1. `requirements.txt` 설치:
```
   pip install -r requirements.txt
```

2. 각 파트의 메인 파일 실행:
```
python main.py  # GUI 실행 (GUI 폴더 안에서 실행)
python main.py   # DB 실행 (DB 폴더 안에서 실행)
python main.py   # CV 실행 (CV 폴더 안에서 실행)
python Robot_arm/main.py  # Robot arm 실행 (보안상 코드 없음)
```

## 주의 사항
1. 통신에 사용되는 IP와 Port 번호를 반드시 확인하세요
``` # 예시: IP 및 Port 설정 확인
HOST = '127.0.0.1'  # 수정할 IP 주소
PORT = 12345        # 수정할 Port 번호
```
2. GUI와 CV에서 사용되는 카메라 및 카메라 번호를 확인하고 설정하세요
``` # 예시: 카메라 번호 확인 및 설정
cap = cv2.VideoCapture(0) # 사용 가능한 카메라 번호로 변경
```

## 시연 영상
시연 영상은 "Demo_video.mp4" 파일로 업로드하였습니다
