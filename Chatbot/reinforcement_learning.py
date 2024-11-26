import gym
import numpy as np
import sqlite3
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sqlite3

# sales_demand_db 데이터베이스 생성
conn = sqlite3.connect('db/sales_demand_db.db')
cursor = conn.cursor()

# sales_actual_vs_predicted 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS sales_actual_vs_predicted (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Flavor TEXT,
    Topping TEXT,
    Predicted_Sales INTEGER,
    Actual_Sales INTEGER
)
''')

# 임시 데이터 삽입
mock_data = [
    ('Banana', 'Lotus', 100, 90),
    ('Banana', 'Rainbow', 120, 110),
    ('Banana', 'Oreo', 130, 125),
    ('Choco', 'Lotus', 95, 100),
    ('Choco', 'Rainbow', 110, 105),
    ('Choco', 'Oreo', 115, 120),
    ('Strawberry', 'Lotus', 80, 75),
    ('Strawberry', 'Rainbow', 90, 85),
    ('Strawberry', 'Oreo', 95, 90)
]

cursor.executemany('''
INSERT INTO sales_actual_vs_predicted (Flavor, Topping, Predicted_Sales, Actual_Sales)
VALUES (?, ?, ?, ?)
''', mock_data)

# 커밋하고 연결 종료
conn.commit()
conn.close()

# inventory 데이터베이스 생성
conn = sqlite3.connect('db/inventory.db')
cursor = conn.cursor()

# stock 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS stock (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Banana INTEGER,
    Choco INTEGER,
    Strawberry INTEGER
)
''')

# 임시 재고 데이터 삽입
cursor.execute('''
INSERT INTO stock (Banana, Choco, Strawberry) 
VALUES (50, 40, 30)
''')

# 커밋하고 연결 종료
conn.commit()
conn.close()

# Action and menu mapping
action_to_menu = {
    0: 'Banana + Lotus',
    1: 'Banana + Rainbow',
    2: 'Banana + Oreo',
    3: 'Choco + Lotus',
    4: 'Choco + Rainbow',
    5: 'Choco + Oreo',
    6: 'Strawberry + Lotus',
    7: 'Strawberry + Rainbow',
    8: 'Strawberry + Oreo'
}

# 판매 예측 데이터 가져오기 (sales_predictions)
def get_sales_predictions():
    conn = sqlite3.connect('db/sales_demand_db.db')  # sales_demand_db 데이터베이스에서 가져오기
    query = '''
    SELECT Flavor, Topping, Predicted_Sales FROM sales_actual_vs_predicted
    '''
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 실제 판매 데이터 가져오기 (sales_demand_db)
def get_sales_data():
    conn = sqlite3.connect('db/sales_demand_db.db')
    query = '''
    SELECT Flavor, Topping, Actual_Sales FROM sales_actual_vs_predicted
    '''
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 재고 데이터 가져오기
def get_stock_data():
    conn = sqlite3.connect('db/inventory.db')  # inventory.db에서 가져오기
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM stock ORDER BY id DESC LIMIT 1')  # 최신 재고 데이터 가져오기
    stock_data = cursor.fetchone()
    conn.close()

    if stock_data:
        return {'Banana': stock_data[1], 'Choco': stock_data[2], 'Strawberry': stock_data[3]}
    return {'Banana': 0, 'Choco': 0, 'Strawberry': 0}  # 기본값 반환

# 강화학습 환경 정의
class IceCreamEnv(gym.Env):
    def __init__(self):
        super(IceCreamEnv, self).__init__()

        # 판매 예측 및 실제 판매 데이터를 가져오기
        self.sales_predictions = get_sales_predictions()
        self.sales_data = get_sales_data()

        # 재고 데이터 초기화
        self.stock = get_stock_data()

        # 액션 공간과 상태 공간 정의
        self.action_space = gym.spaces.Discrete(9)  # 9개의 액션 (메뉴 * 토핑)
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(3,), dtype=np.int32)  # 재고 상태

    def reset(self):
        # 초기 재고 데이터로 리셋
        self.stock = get_stock_data()
        return np.array(list(self.stock.values()))  # 상태 반환

    def step(self, action):
        menu = action_to_menu[action]
        reward = 0

        # 판매 예측에 따른 보상 계산
        flavor, topping = menu.split(' + ')
        predicted_sales = self.sales_predictions[
            (self.sales_predictions['Flavor'] == flavor) &
            (self.sales_predictions['Topping'] == topping)
        ]['Predicted_Sales'].values[0]

        actual_sales = self.sales_data[
            (self.sales_data['Flavor'] == flavor) &
            (self.sales_data['Topping'] == topping)
        ]['Actual_Sales'].values[0]

        # 예측된 판매량과 실제 판매량의 차이를 보상으로 설정
        sales_difference = abs(predicted_sales - actual_sales)
        reward = -sales_difference  # 차이가 클수록 보상이 적어짐

        # 재고 업데이트
        if flavor == 'Banana' and self.stock['Banana'] > 0:
            self.stock['Banana'] -= 1
        elif flavor == 'Choco' and self.stock['Choco'] > 0:
            self.stock['Choco'] -= 1
        elif flavor == 'Strawberry' and self.stock['Strawberry'] > 0:
            self.stock['Strawberry'] -= 1
        else:
            reward = -1  # 재고 부족 시 보상 감소

        next_state = np.array(list(self.stock.values()))  # 새로운 재고 상태
        done = all(value == 0 for value in self.stock.values())  # 모든 재고가 소진되면 종료
        return next_state, reward, done, {}

    def render(self):
        # 현재 재고 상태 출력
        print(f"Current stock: {self.stock}")

# 환경 초기화
env = IceCreamEnv()

# 로그를 위한 변수 초기화
actions = []  # 액션
rewards = []  # 보상
states = []  # 상태
stock_log = []  # 재고 로그

# 훈련 루프
for episode in range(2):  # 여러 에피소드 반복
    state = env.reset()  # 환경 초기화
    done = False
    while not done:
        action = env.action_space.sample()  # 랜덤 액션 선택
        next_state, reward, done, _ = env.step(action)

        # 액션, 보상, 상태, 재고 기록
        actions.append(action)
        rewards.append(reward)
        states.append(state)
        stock_log.append(dict(zip(env.stock.keys(), next_state)))  # 현재 재고 상태 기록

        state = next_state  # 상태 업데이트

# 3D 그래프 시각화
x = np.array(actions)  # 액션을 숫자로
y = np.array(rewards)  # 보상
z = np.arange(len(actions))  # 시간 단계 (액션 순서)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', marker='o', label='Actions-Rewards-Time')

ax.set_xlabel('Actions (Menu + Topping)')
ax.set_ylabel('Rewards')
ax.set_zlabel('Time Steps')
ax.set_title('3D Visualization of Reinforcement Learning')

# x축에 문자열 레이블 매핑
ax.set_xticks(np.arange(9))  # 0부터 8까지 숫자 표시
ax.set_xticklabels([action_to_menu[i] for i in range(9)], rotation=45, ha='right')

# 레전드 추가
ax.legend()

# 그래프 저장
img_folder = 'img'
if not os.path.exists(img_folder):
    os.makedirs(img_folder)
plt.savefig(f'{img_folder}/reinforcement_learning_3d.png')
plt.show()

# 데이터프레임 생성 및 저장
df = pd.DataFrame({
    'Actions': [action_to_menu[action] for action in actions],
    'Rewards': rewards,
    'States': states,
    'Stock_Log': stock_log
})

df_folder = 'df'
df_folder = './df'
if not os.path.exists(df_folder):
    os.makedirs(df_folder)
df.to_csv(f'{df_folder}/reinforcement_learning_data.csv', index=False)


print("3D 그래프와 데이터프레임이 저장되었습니다!", flush=True)

# 데이터베이스에서 학습
