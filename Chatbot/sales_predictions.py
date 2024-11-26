import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 플로팅을 위한 추가 모듈
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
import sqlite3

# 예시 데이터: is_match_day, temperature, sales
data = np.array([
    [1, 20, 100],  # 경기 있음, 온도 20도, 판매량 100
    [0, 25, 200],  # 경기 없음, 온도 25도, 판매량 200
    [1, 22, 150],  # 경기 있음, 온도 22도, 판매량 150
    [0, 18, 130],  # 경기 없음, 온도 18도, 판매량 130
    [1, 30, 250],  # 경기 있음, 온도 30도, 판매량 250
    [0, 15, 90],   # 경기 없음, 온도 15도, 판매량 90
])

# 특성(X)와 타겟(y) 분리
X = data[:, :2]  # is_match_day, temperature
y = data[:, 2]   # sales (판매량)

# MinMaxScaler를 사용하여 데이터 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 형태 변환: (배치 크기, 시간 단계, 특성 수)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# GRU 모델 정의
model = Sequential([
    GRU(50, activation='relu', input_shape=(1, 2)),  # 1은 time_steps, 2는 features
    Dense(1)  # 예측 값 (판매량)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 모델 훈련
model.fit(X_reshaped, y, epochs=100, batch_size=2)

# 예측: 전체 데이터 예측
predictions = model.predict(X_reshaped)

# 예측된 판매량과 실제 판매량을 데이터프레임으로 저장
df = pd.DataFrame({
    'is_match_day': data[:, 0],
    'temperature': data[:, 1],
    'actual_sales': y,
    'predicted_sales': predictions.flatten()
})

# 테스트 데이터 예시: 경기 있는 날, 온도 22도
test_data = np.array([[1, 22]])  # 경기 있는 날, 온도 22도
test_data_scaled = scaler.transform(test_data)  # 테스트 데이터 스케일링

# 데이터 형태 변환
test_data_reshaped = test_data_scaled.reshape((test_data_scaled.shape[0], 1, test_data_scaled.shape[1]))

# 예측
predicted_sales = model.predict(test_data_reshaped)

# 예측 결과 출력
print(f"입력된 테스트 데이터: 경기 여부: {test_data[0][0]}, 온도: {test_data[0][1]}도")
print(f"예측된 판매량: {predicted_sales[0][0]}")

# 3D 그래프 생성
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 실제 판매량을 3D로 플로팅
ax.scatter(df['is_match_day'], df['temperature'], df['actual_sales'], color='b', label='Actual Sales', marker='o')

# 예측된 판매량을 3D로 플로팅
ax.scatter(df['is_match_day'], df['temperature'], df['predicted_sales'], color='r', label='Predicted Sales', marker='x')

# 제목과 레이블 설정
ax.set_title("3D Actual vs Predicted Sales")
ax.set_xlabel("Match Day (0: No, 1: Yes)")
ax.set_ylabel("Temperature (°C)")
ax.set_zlabel("Sales")

# 레전드 추가
ax.legend()

# 이미지 저장할 경로 지정
img_folder = 'img'
if not os.path.exists(img_folder):  # img 폴더가 없으면 생성
    os.makedirs(img_folder)

# 그래프 이미지를 'img' 폴더에 저장
plt.savefig(f'{img_folder}/3d_sales_prediction.png')

# 그래프 화면에 표시
plt.show()

# df 폴더가 없으면 생성
df_folder = 'df'
if not os.path.exists(df_folder):
    os.makedirs(df_folder)

# 예측 결과를 'df' 폴더에 CSV로 저장
df.to_csv(f'{df_folder}/sales_predictions.csv', index=False)

# SQLite 데이터베이스에 저장
db_folder = 'db'
if not os.path.exists(db_folder):
    os.makedirs(db_folder)

# SQLite 데이터베이스 파일 경로
db_path = f'{db_folder}/sales_predictions.db'

# SQLite 연결 및 테이블 생성
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 테이블 생성 (존재하지 않으면)
cursor.execute('''
CREATE TABLE IF NOT EXISTS sales_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    is_match_day INTEGER,
    temperature REAL,
    actual_sales INTEGER,
    predicted_sales REAL
)
''')

# DataFrame의 각 행을 데이터베이스에 삽입
for index, row in df.iterrows():
    cursor.execute('''
    INSERT INTO sales_predictions (is_match_day, temperature, actual_sales, predicted_sales)
    VALUES (?, ?, ?, ?)
    ''', (row['is_match_day'], row['temperature'], row['actual_sales'], row['predicted_sales']))

# 변경 사항 커밋 후 연결 종료
conn.commit()
conn.close()

# 예측 결과 출력
print("\n예측 결과 데이터프레임:")
print(df)
