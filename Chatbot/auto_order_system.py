import pandas as pd
from datetime import datetime
import ast  # ast 모듈을 임포트

# 데이터프레임으로 데이터를 생성합니다.

# 강화학습 데이터 (예시)
rl_data = [
    ["Strawberry + Lotus", -171.23, "[50, 40, 30]", "{'Banana': 50, 'Choco': 40, 'Strawberry': 29}"],
    ["Banana + Choco", -100.12, "[20, 30, 40]", "{'Banana': 30, 'Choco': 30, 'Strawberry': 25}"]
]
df_rl = pd.DataFrame(rl_data, columns=["Actions", "Rewards", "States", "Stock_Log"])

# 재고 데이터 (예시)
inventory_data = [
    ["Banana", 100],
    ["Choco", 50],
    ["Strawberry", 30],
    ["Lotus", 20]
]
df_inventory = pd.DataFrame(inventory_data, columns=["product_name", "stock_quantity"])

# 주문 데이터 (예시)
order_data = [
    ["Banana", 10, "2024-11-21 12:00:00"],
    ["Choco", 5, "2024-11-21 12:05:00"]
]
df_orders = pd.DataFrame(order_data, columns=["product_name", "quantity", "order_date"])

# 'States' 컬럼의 값을 파싱하여 리스트로 변환하는 함수
def parse_states(state_str):
    """
    'States' 컬럼의 값을 공백을 쉼표로 바꾸어 리스트로 변환하고,
    빈 값이 있는 경우 이를 필터링하여 처리합니다.
    """
    if isinstance(state_str, str):
        state_str = state_str.replace(' ', ',')
        state_list = [x for x in state_str.strip("[]").split(',') if x]
    elif isinstance(state_str, list):
        state_list = state_str
    else:
        state_list = []

    return state_list

# 강화학습 데이터에서 추천 액션과 결과를 가져오는 함수
def fetch_rl_suggestions():
    """
    강화학습 데이터에서 추천 액션과 결과를 가져옵니다.
    """
    # 'States' 컬럼의 값들을 파싱하여 리스트 형식으로 변환
    df_rl['States'] = df_rl['States'].apply(parse_states)

    # 'Stock_Log' 컬럼이 문자열일 경우, 안전하게 딕셔너리로 변환
    def safe_eval(stock_log):
        try:
            return ast.literal_eval(stock_log) if isinstance(stock_log, str) else stock_log
        except (ValueError, SyntaxError):
            return {}

    df_rl['Stock_Log'] = df_rl['Stock_Log'].apply(safe_eval)

    # 데이터프레임에서 필요한 컬럼만 추출하여 리스트로 반환
    suggestions = []
    for _, row in df_rl.iterrows():
        suggestions.append({
            'action': row['Actions'],
            'state': row['States'],
            'stock_log': row['Stock_Log']
        })
    return suggestions

# 재고 데이터에서 현재 재고 상태를 가져오는 함수
def fetch_inventory():
    """
    재고 데이터를 가져옵니다.
    """
    inventory_data = {row['product_name']: row['stock_quantity'] for _, row in df_inventory.iterrows()}
    return inventory_data

# 주문 데이터에 새로운 주문을 추가하는 함수
def add_order(product_name, quantity):
    """
    주문 데이터에 새로운 주문을 추가합니다.
    """
    order_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_order = pd.DataFrame({
        'product_name': [product_name],
        'quantity': [quantity],
        'order_date': [order_date]
    })
    global df_orders
    df_orders = pd.concat([df_orders, new_order], ignore_index=True)
    print(f"자동 주문이 추가되었습니다: {product_name} x {quantity}")

# 자동 주문 생성 함수
def auto_order():
    """
    강화학습 데이터와 재고 데이터를 이용해 자동 주문을 생성합니다.
    """
    print("자동 주문을 시작합니다...")

    rl_suggestions = fetch_rl_suggestions()
    inventory_data = fetch_inventory()

    for suggestion in rl_suggestions:
        action = suggestion['action']
        stock_log = suggestion['stock_log']

        items = action.split(' + ')
        for item in items:
            product_name = item
            if product_name in inventory_data:
                current_stock = inventory_data[product_name]
                recommended_stock = stock_log.get(product_name, current_stock)

                if current_stock < recommended_stock:
                    order_quantity = recommended_stock - current_stock
                    add_order(product_name, order_quantity)

    print("자동 주문이 완료되었습니다.")

if __name__ == '__main__':
    # 자동 주문 실행
    auto_order()

    # 주문 데이터 출력
    print(df_orders)

