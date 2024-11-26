import socket
import time
import threading
import datetime
from ultralytics import YOLO
import cv2

# 필요한 기능 별로 클래스를 만들어서 각 기능을 수행하게 하면 좋을 듯
# ex. 충돌 감지 클래스 / 고객 인식 클래스 / 로봇 동작 지원 클래스

# Robot과의 통신을 위한 클래스
class Robot_Connect():
    def __init__(self):
        super().__init__()


    def connect_robot(self):
        # socket 연결 with robot arm (연결 될때까지 지속적 시도.)
        self.HOST = '127.0.0.1'  # 본인 컴퓨터의 IP 입력
        self.PORT = 43210           
        self.BUFSIZE = 1024
        self.ADDR = (self.HOST, self.PORT)

        # 서버 소켓 설정
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serverSocket.settimeout(10)

        # 연결 시도
        while True:
            try:
                self.serverSocket.bind(self.ADDR)

                # 연결 대기
                self.serverSocket.listen(1)
                print(f'[Robot] 서버와 연결 시도 중 입니다.')
                time.sleep(1)
                while True:
                    try:
                        self.clientSocket, addr_info = self.serverSocket.accept()
                        print("[Robot] 서버와 연결 성공")
                        break
                    except socket.timeout:
                        print("[Robot] socket timeout")
                        print("[Robot] 서버와 연결을 재시도 합니다")
                break
            except:
                pass

        print("--client info--")
        print(self.clientSocket)

        self.connected = True
        
        # 지속적으로 메시지 수령.
        while self.connected:
            print("[Robot] 메시지 수신 대기중")
            time.sleep(0.5)
            
            try:
                # 대기 시간 설정
                self.clientSocket.settimeout(20.0)
                # 메시지 수령
                self.recv_msg = self.clientSocket.recv(self.BUFSIZE).decode('utf-8')
                print('[Robot] received msg : ' + self.recv_msg)
                
                
                # 메시지가 비어 있는 경우. 연결이 끊겼으므로 재연결을 위해 예외 처리
                if self.recv_msg == '':
                        print('[Robot] received empty msg')
                        raise Exception('empty msg')

                self.recv_msg = self.recv_msg.split('/')

                ############################################################################
                # 받은 요청에 따라 응답해주는 부분.

                # request 타입이라면 (어떤 비전 작업을 요청)
                if self.recv_msg[0] == 'jig_check':
                    # jig 확인 요청을 받고 의도된 jig에 아이스크림 캡슐을 제대로 배치되었는지 확인
                    jig_state = cam_table_object_detection.jig_check(self.recv_msg[1])
                    self.clientSocket.send(("jig_check/" + jig_state).encode('utf-8'))
                    # self.clientSocket.send(("jig_check/go").encode('utf-8'))                    # test 코드
                    # self.clientSocket.send(("jig_check/stop").encode('utf-8'))                    # test 코드
                if self.recv_msg[0] == 'seal_check':
                    # self.clientSocket.send(("seal_check/go").encode('utf-8'))                   # test 코드
                    # self.clientSocket.send(("seal_check/stop").encode('utf-8'))                   # test 코드
                    # seal 제거 여부 확인 요청을 받고 seal의 존재 여부를 확인
                    seal_state = cam_table_object_detection.seal_check()
                    self.clientSocket.send(("seal_check/"+seal_state).encode('utf-8'))  
                if self.recv_msg[0] == 'customer_check':
                    # cam_front_customer_thread.start()
                    # time.sleep(1)
                    # 고객의 위치 정보를 요청 받고 고객의 위치 정보를 확인
                    # self.clientSocket.send(("customer_check/y/290,0").encode('utf-8'))        # test 코드
                    # self.clientSocket.send(("customer_check/n").encode('utf-8'))        # test 코드
                    customer_state = cam_front_object_detection.customer_check()
                    self.clientSocket.send(("customer_check/"+customer_state).encode('utf-8'))
                    # cam_front_object_detection.cam_front_flag = True
                    # cam_front_customer_thread.join() 
                if self.recv_msg[0] == 'receive_check':     # 고객이 직접 수령할 때 수령 여부 확인
                    # cam_front_receive_thread.start()
                    # time.sleep(1)
                    # self.clientSocket.send(("receive_check/y").encode('utf-8'))                 # test 코드
                    # self.clientSocket.send(("receive_check/n").encode('utf-8'))                 # test 코드
                    receive_state = cam_front_object_detection.receive_check()
                    self.clientSocket.send(("receive_check/"+receive_state).encode('utf-8'))
                    # cam_front_object_detection.cam_front_flag = True
                    # cam_front_receive_thread.join()
                if self.recv_msg[0] == 'jig_receive_check': # JIG에 서빙 후 고객의 수령 여부 확인
                    # self.clientSocket.send(("jig_receive_check/y").encode('utf-8'))             # test 코드
                    while True:
                        jig_receive_state = cam_table_object_detection.jig_receive_check(self.recv_msg[1])          # 비동기 실행 해야하나? 겹치면 문제 될수도.
                        self.clientSocket.send(("jig_receive_check/"+jig_receive_state).encode('utf-8')) 
                        if jig_receive_state == 'y':
                            break


                if self.recv_msg[0] == 'icecream_service_start':
                    cam_table_thread = threading.Thread(target=cam_table_object_detection.object_detection)
                    if not cam_table_thread.is_alive():
                        cam_table_thread.start()
                    cam_front_thread = threading.Thread(target=cam_front_object_detection.object_detection)
                    if not cam_front_thread.is_alive():
                        cam_front_thread.start()
                if self.recv_msg[0] == 'icecream_service_finish':
                    pass
                    # cam_table_object_detection.cam_table_flag = True            # Cam Table object detection thread 종료.
                    # cam_front_object_detection.cam_front_flag = True
                ############################################################################
                # else:
                #     # 예상되지 않은 메시지를 받은 경우
                #     self.clientSocket.send('ERROR : wrong msg received'.encode('utf-8'))
                #     print('got unexpected msg!')
                
            except socket.timeout:
                print('[Robot] MainException: {}'.format(socket.timeout))

            except Exception as e:
                print('[Robot] MainException: {}'.format(e))
                self.connected = False
                print('[Robot] connection lost')
                # 재연결 시도
                while True:
                    time.sleep(2)
                    try:
                        # server socket 정리
                        self.serverSocket.shutdown(socket.SHUT_RDWR)
                        self.serverSocket.close()
                        
                        # 소켓 설정
                        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        self.serverSocket.settimeout(10)

                        while True:
                            self.serverSocket.bind(self.ADDR)
                            self.serverSocket.listen(1)
                            print(f'[Robot] 재연결 시도 중 입니다')
                            try:
                                self.clientSocket, addr_info = self.serverSocket.accept()
                                self.connected = True
                                break

                            except socket.timeout:
                                print('[Robot] socket.timeout')

                            except:
                                pass
                        break
                    except Exception as e:
                        print('[Robot] MainException: {}'.format(e))

# 
class DB_Connect():
    def __init__(self):
        super().__init__()


    def connect_db(self):
        self.ADDR = '127.0.0.1'          # DB 주소로 변경
        self.PORT = 23456                   # DB의 Port로 변경
        
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        while True:
            
            print("[DB] 서버와 연결 시도 중 입니다.")
            try:
                self.client_socket.connect((self.ADDR, self.PORT))
                self.connected = True
                break
            except socket.error as e:
                print(f"[DB] 서버에 연결할 수 없습니다: {e}")
                print("[DB] 2초 후 연결을 재시도 합니다")
                time.sleep(2.0)

        while self.connected:
            
            time.sleep(1)
            print("[DB] 메시지 수신 대기중")
            try:
                # 대기 시간 설정
                self.client_socket.settimeout(10.0)
                # 메시지 수령
                self.recv_msg = self.client_socket.recv(1024).decode('utf-8')
                
                # 메시지가 비어 있는 경우. 연결이 끊겼으므로 재연결을 위해 예외 처리
                if self.recv_msg == '':
                    print("[DB] received empty msg")
                    raise Exception("empty msg")
                
                print("[DB] receieved msg : "+ self.recv_msg)

                #########################################
                # 전달받는 메시지 처리부

                # 메시지 전처리
                msg_list = self.recv_msg.split('\n')[-2]
                msg_list = msg_list.split('/')
                if self.last_msg_time != msg_list[1]:
                    if msg_list[0] == 'db_error':
                        print("[DB] 데이터 처리 중 에러 발생")
                    elif msg_list[0] == 'incident_saved': #######기존 가입한 회원
                        print("[DB] 사고기록 저장완료:,",msg_list)
                    elif msg_list[0]=="is_table_clean":
                        print("[DB] table 상태 확인요청")
                        cam_table_object_detection.is_dirty()
                    elif msg_list[0] == 'test': 
                        print("[DB] test")
                    else:
                        print("[DB] 사전에 정의되지 않은 메시지 형식")
                        print(msg_list)
                else:
                    #print("[from DB]이전과 같은 메시지수신")
                    data = 'test/0\n' 
                    self.client_socket.sendall((data).encode('utf-8'))

                self.last_msg_time = msg_list[1]
                #########################################

            except socket.timeout:
                print('[DB] MainException: {}'.format(socket.timeout))
            except Exception as e:
                print('[DB] MainException: {}'.format(e))
                self.connected = False
                print('[DB] connection lost')
                # 재연결 시도
                while True:
                    
                    time.sleep(2)
                    try:
                        # 소켓 정리
                        self.client_socket.shutdown(socket.SHUT_RDWR)
                        self.client_socket.close()
                        
                        # 소켓 설정
                        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                        try:
                            self.client_socket.connect((self.ADDR, self.PORT))
                            self.connected = True
                        except socket.error as e:
                            print("[DB] 서버에 연결할 수 없습니다. 2초 후 재시도합니다.")
                        
                    except Exception as e:
                        print('[DB] MainException: {}'.format(e))
                        print('[DB] except')
                        # pass

                        # DB에게 데이터를 전송하는 함수를 짤 때 self.connected를 확인하면 좋을 듯

class Cam_Table_Object_Detection():           # 테이블 및 로봇암을 바라보는 웹캠
    def __init__(self):
        super().__init__()
        self.results = None
        self.cam_table_flag = False

        self.model_weight_path = './cam_table_1119_best.pt'
        self.model = YOLO(self.model_weight_path, verbose=False)

        self.cls_model_weight_path = './cam_table_clean_1119_best2.pt'
        self.cls_model = YOLO(self.cls_model_weight_path, verbose=False)
        self.table_is_clean = True #청결한 상태라고 가정하고 시작

        self.time_slot = 0.1        # Object detection을 몇초마다 진행할 것인지.
        self.last_processed_time_is_coll =datetime.datetime.now() # 충돌계산, n초단위로 신호전송목적

        self.robot_stop = False                         # 로봇이 현재 중단중인지 기록
        self.robot_resume_time = 0.0                    # 로봇 재개를 위해 collision이 발생하지 않은 시간을 기록
        
        self.robot_arm_xyxy = []
    """
        cam1(table방향)
        1. 탐지대상
            - 로봇암
            - 손
            - 아이스크림캡슐
                - 캡슐
                - 씰
        2. Task
            - [신호시, 1번]Jig 확인 => 아이스크림 캡슐(2번 class) 
            - [신호시, 1번]씰 제거확인 => 아이스크림 씰(3번 class)
            - [자동, 1초에 1번]충돌확인 => 로봇암(0), 손(1)
            
    """

    @staticmethod
    def calculate_distance(center1, center2):
        """두 점 사이의 유클리드 거리 계산"""
        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    

    def object_detection(self):
        cap = cv2.VideoCapture(2)  # 0은 기본 웹캠을 의미. 다른 웹캠을 사용할 경우 번호를 변경

        if not cap.isOpened():
            print("Failed to open the webcam(cam_table).")
        else:
            while True:
                if self.cam_table_flag:         # 종료 신호가 온다면
                    break

                ret, frame = cap.read()
                self.frame = frame
                if not ret:
                    print("Failed to grab frame.")
                    break

                result = self.model.predict(source=frame, conf=0.25, verbose=False)
                self.results = result
                
                # Display the prediction result######################################
                predicted_frame = result[0].plot()
                self.predicted_frame = predicted_frame
                # cv2.imshow('YOLOv8 Real-time Prediction', predicted_frame)
                #####################################################################
                # cv2.waitKey(1)

                # 충돌감지 trigger-손등장 를 위한 탐지결과 해석
                detected_class_list=[0,0,0,0,0]
                if result != None:
                    for i in range(len(result[0])):
                        class_num = int(result[0][i].boxes.cls) 
                        if class_num == 0: #로봇암
                            detected_class_list[0] +=1              
                        elif class_num == 1: #손
                            detected_class_list[1] +=1
                            print("Hand_detected")
                        elif class_num == 2: #아이스크림 캡슐
                            detected_class_list[2] +=1
                        elif class_num == 3: #씰
                            detected_class_list[3] +=1
                        else: # 기타 무언가?
                            detected_class_list[4] +=1
                            # print("Class 4 detected-4번이 어떤객체인가요")

                    if detected_class_list[1] >=1 :
                        # self.is_collision(process_interval=3, threshold=-100)
                        self.robot_resume_time = 0
                    else:
                        if self.robot_stop == True:
                            self.robot_resume_time += self.time_slot
                
                if self.robot_stop == True:
                    if self.robot_resume_time > 5.0:
                        try:
                            print("위험 요소 제거 후 5초가 지났으므로 로봇 행동 재개 신호 전송")
                            robot_connect.clientSocket.send("robot_resume".encode('utf-8'))
                            self.robot_stop = False
                        except:
                            print("로봇 행동 재개 신호 전송 실패!")


                time.sleep(self.time_slot)

        self.cam_table_flag = False
        # 웹캠과 창 닫기
        cap.release()
        cv2.destroyAllWindows()


    def jig_check(self, jig):        # 매개변수로 전달받은 특정 JIG에 아이스크림 캡슐이 올바르게 배치되었는지 확인 후 결과 반환
        jig_positions = {                  # jig A, B, C의 중심 좌표
            'A': (589, 46),
            'B': (452, 45),
            'C': (304, 46)
        }
        jig_threshold = 20  # threshold 값 설정 (기본값 50 픽셀, icecream_capsule 중심 좌표와 jig 중심 좌표간의 픽셀 거리)

        jig_state = 'stop'  # 초기 상태는 'stop'
        jig_center = jig_positions.get(jig, None)

        if not jig_center:
            print(f"Invalid jig name: {jig}")
            return 'stop'

        time_in_position = 0  # icecream_capsule이 jig에 있는 시간을 추적
        
        # time limit 설정 (10초)
        time_limit = 10.0  # 10초
        total_time = 0.0  # 총 경과 시간을 추적

        while True:
            for result in self.results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    if self.model.names[class_id] == 'icecream_capsule':  # icecream_capsule만 확인
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                        # 중앙 좌표 계산
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # 중앙 좌표와 jig의 중앙 좌표 간의 거리 계산
                        distance = self.calculate_distance((center_x, center_y), jig_center)

                        # threshold 값을 기준으로 'go' 여부 판단
                        if distance < jig_threshold:  # threshold 값 내에 있으면 'go'
                            time_in_position += self.time_slot  # icecream_capsule이 jig에 머무른 시간을 추가
                        else:
                            time_in_position = 0  # 떨어지면 시간 초기화

            # 5초 동안 jig에 올바르게 배치된 상태로 유지되면 'go'
            if time_in_position >= 5.0:  # 50이라는 숫자는 약 5초에 해당
                jig_state = 'go'
                print(f"Capsule placed correctly on {jig}!")
                break   
            
            # time limit 체크: 10초가 지나면 stop으로 설정
            total_time += self.time_slot  # 경과 시간 증가
            if total_time >= time_limit:  # Time limit 넘길 경우
                print(f"Time limit of {time_limit} seconds reached, stopping.")
                jig_state = 'stop'
                break

            time.sleep(self.time_slot)  # 정해진 시간마다 상태를 체크

        return jig_state
    
    def seal_check(self):
    
        time_limit = 10.0  # 최대 실행 시간 (초)
        total_time = 0.0  # 총 경과 시간
        seal_state = 'stop'  # 초기 상태는 'stop'

        # 특정 영역에서만 감지하기 위한 ROI 영역 정의 (x_min, y_min, x_max, y_max)-> ###roi 좌표 수정 필요함
        roi = (400, 100, 640, 480)

        # 감지된 프레임 수 추적
        seal_detected_frame_count = 0
        seal_star_detected_frame_count = 0
        nothing_detected_frame_count = 0

        while total_time < time_limit:
            if self.results is not None:
                results = self.results

                # 매 프레임마다 초기화
                seal_found = False
                seal_star_found = False

                # 바운딩 박스를 순회하며 객체 탐지
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # 바운딩 박스 좌표와 클래스 정보 가져오기
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]

                        # ROI 내부의 객체만 처리
                        if (roi[0] <= x_min <= roi[2] and roi[0] <= x_max <= roi[2] and 
                        roi[1] <= y_min <= roi[3] and roi[1] <= y_max <= roi[3]):
                            if class_name == 'seal':
                                seal_found = True
                            elif class_name == 'seal_star':
                                seal_star_found = True

                # 감지 및 상태 추적
                if seal_star_found:
                    seal_star_detected_frame_count += 1
                    if seal_star_detected_frame_count >= 20:
                        seal_state = 'go'
                        print("seal_star detected for 20 frames. Seal state updated to: 'go'")
                        break

                if seal_found:
                    seal_detected_frame_count += 1
                    if seal_detected_frame_count >= 20:
                        seal_state = 'stop'
                        print("seal detected for 20 frames. Seal state updated to: 'stop'")
                        break

            # 아무것도 감지되지 않은 경우
            if not seal_found and not seal_star_found:
                nothing_detected_frame_count += 1
                if nothing_detected_frame_count >= 20:
                    seal_state = 'go'
                    print("No detection for 20 frames. Seal state updated to: 'go'")
                    break
                

            total_time += self.time_slot  # self.time_slot로 시간 간격 추가
            time.sleep(self.time_slot)  # self.time_slot 간격만큼 대기

        # Time limit 동안 감지가 유지되거나 10프레임 조건이 충족되지 않은 경우
        print(f"Time limit reached. Final seal state: '{seal_state}'")
        return seal_state

    """
    def seal_check(self):
        # `seal_check`에서만 사용될 설정
        time_limit = 10.0  # 10초 동안 확인할 수 있도록 설정 (10초)
        seal_recognition_time = 3.0  # seal_star가 인식되어야 할 최소 시간 (3초)

        seal_state = 'stop'  # 초기 상태는 'stop'
        time_in_position = 0.0  # seal_star가 인식된 시간을 추적
        total_time = 0.0  # 전체 경과 시간을 추적

        if self.results != None:
            while total_time < time_limit:
                results = self.results
                seal_star_found = False

                # 바운딩 박스를 찾고 seal_star가 인식되었는지 확인
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        if self.model.names[class_id] == 'seal_star':  # seal_star만 확인
                            seal_star_found = True
                            break

                if seal_star_found:
                    time_in_position += self.time_slot  # seal_star가 인식된 시간을 추가
                else:
                    time_in_position = 0  # seal_star가 인식되지 않으면 시간 초기화

                # 3초 동안 seal_star가 인식되면 'go'
                if time_in_position >= seal_recognition_time:  # 3초는 약 30프레임
                    seal_state = 'go'
                    print("Seal star is recognized for 3 seconds!")
                    break

                total_time += self.time_slot  # 전체 시간 경과 (프레임 단위)
                time.sleep(self.time_slot)  # 100ms마다 상태를 체크

        if total_time >= time_limit and seal_state == 'stop':
            print("Seal star was not recognized in 10 seconds.")

        return seal_state
    """

    def jig_receive_check(self, jig):
        jig_positions = {
            'A': (591, 46),
            'B': (450, 45),
            'C': (304, 45)
        }
        jig_receive_threshold = 80
        # 수령 여부 확인
        jig_receive_state = 'n'  # 초기 상태는 'n'
        jig_center = jig_positions.get(jig, None)
        roi = (235, 0, 640, 140)
        if not jig_center:
            print(f"Invalid jig name: {jig}")
            return 'n'
        time_limit = 10.0
        total_time = 0.0  # 총 경과 시간을 추적
        # jig_hold_time = 0.0  # icecream_cup이 jig 근처에서 감지되지 않거나 멀리 떨어져 있는 상태가 5초간 유지되는지 추적
        cup_lost_time = 0.0  # icecream_cup이 사라진 시간을 추적
        while total_time < time_limit:  # 10초 동안 확인 (100ms마다 체크)
            icecream_cup_found = False
            for result in self.results:             # 컵이 감지된 경우
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    if self.model.names[class_id] == 'icecream_cup':  # icecream_cup만 확인
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                        if (roi[0] <= x1 <= roi[2] and roi[0] <= x2 <= roi[2] and roi[1] <= y1 <= roi[3] and roi[1] <= y2 <= roi[3]):
                        # 중앙 좌표 계산
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            # 중앙 좌표와 jig의 중앙 좌표 간의 거리 계산
                            distance = self.calculate_distance((center_x, center_y), jig_center)
                            # icecream_cup이 jig의 중심 근처에서 감지되지 않거나, jig_threshold 보다 멀리 떨어져 있는 경우
                            if distance > jig_receive_threshold:
                                cup_lost_time += self.time_slot  # 5초 동안 멀어진 상태를 유지하는지 추적
                                print(f"Distance between icecream_cup and jig: {distance} (Threshold: {self.jig_threshold})")
                            else:
                                cup_lost_time = 0  # 두 객체가 가까워지면 상태 초기화
                            icecream_cup_found = True
            # icecream_cup이 5초 이상 감지되지 않으면 'y' 전송
            if not icecream_cup_found:              # 컵이 감지 안되는 경우
                cup_lost_time += self.time_slot
                print(f"Ice cream cup not detected.")
            # 5초 이상 멀어졌으면 'y' 전송
            if cup_lost_time >= 3.0:  # 5초 이상 (100ms 기준으로 50)
                jig_receive_state = 'y'
                print(f"Jig ice cream taken, sending 'y'.")
                break
            total_time += self.time_slot # 경과 시간 증가
            time.sleep(self.time_slot)  # 100ms마다 상태를 체크
        # 시간 제한을 초과하면 'n' 전송
        if total_time >= time_limit and jig_receive_state == 'n':
            print(f"Time limit of {time_limit} seconds reached, no ice cream received. Sending 'n'.")
            jig_receive_state = 'n'
        return jig_receive_state
    
    def is_collision(self, process_interval=3, threshold=0):
        """ 
        로봇암과 손의 충돌 여부 확인 목적
        parameter인 results는 아래에서의 모델 추론결과로 받는 객체를 의미함
            model = YOLO(yolo_weight_path)
            results = model.predict(image_source)
        
        [threshold]의 경우 음수가 될수록 더 넓은 영역이 겹쳐야 충돌로 판단하게 됨, 0은 박스가 맞닿는 경우 충돌로 판단함을 의미함

        1. self.is_collision_state 갱신 (True: 충돌로 판단, False: 충돌은 아님)
        2. 최종적으로 아래 형식으로 DB에게 메시지 전송
            incident/'+ str(충돌시간)+'/'+str(max_iou값)+'/'+'사람과의충돌여부(현재는 손에대해서만 하므로 항상 True)'+'/'+str(로봇암이 멈췄는지)+'/'+str(기록물이 있다면 보관위치)+'\n'
        
        """
        # result에서 xyxy정보 분리
        hand_xyxy = []
        cap_xyxy=[]
        results = self.results
        collision_time = ""

        if results != None:
            for i in range(len(results[0])):
                class_num = int(results[0][i].boxes.cls) 
                xyxy_list = results[0][i].boxes.xyxy.tolist()[0]
                if class_num == 0:
                    self.robot_arm_xyxy=xyxy_list
                elif class_num == 1:
                    hand_xyxy.append(xyxy_list)
                else:
                    cap_xyxy.append(xyxy_list)
       
            # 충돌계산       
            if not isinstance(hand_xyxy[0], list):
                hand_xyxy = [hand_xyxy]  # 단일 손 좌표를 리스트로 변환
        
            max_iou = 0.0
            
            for idx, hand_box in enumerate(hand_xyxy):
                hand_x1, hand_y1, hand_x2, hand_y2 = hand_box
                robot_x1, robot_y1, robot_x2, robot_y2 = self.robot_arm_xyxy
                
                # 겹치는 영역 계산
                x_left = max(hand_x1, robot_x1)
                y_top = max(hand_y1, robot_y1)
                x_right = min(hand_x2, robot_x2)
                y_bottom = min(hand_y2, robot_y2)
                
                # threshold를 적용한 겹침 확인
                iou = 0.0
                
                if (x_right + threshold >= x_left - threshold and 
                    y_bottom + threshold >= y_top - threshold):                         # 충동했다고 판단함
                    
                    # 로봇 중단 시키기.
                    if self.robot_stop == False:                                        # 로봇이 현재 중단 상태가 아니라면
                        try:
                            print("충돌 발생 로봇 정지 신호 전송")
                            robot_connect.clientSocket.send("robot_pause".encode('utf-8'))  # 중단 신호 전송
                            self.robot_stop = True                                          # 로봇 중단 상태 기록
                        except:
                            print("로봇 일시정지 신호 전송 실패!")

                    # 충돌시간 기록
                    collision_time = datetime.datetime.now()
                    
                    # 겹치는 영역의 넓이 계산
                    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
                    
                    # 각 박스의 넓이 계산
                    hand_area = (hand_x2 - hand_x1) * (hand_y2 - hand_y1)
                    robot_area = (robot_x2 - robot_x1) * (robot_y2 - robot_y1)
                    
                    # 합집합 넓이 계산
                    union_area = hand_area + robot_area - intersection_area
                    
                    # IoU (Intersection over Union) 계산
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    if iou > max_iou:
                        max_iou = round(iou,2)

                    time_difference = collision_time - self.last_processed_time_is_coll
                    time_difference = float(time_difference.total_seconds() )

                    if time_difference >= process_interval:    
                        try:
                            data = 'incident/'+ str(collision_time)+'/'+str(max_iou)+'/'+'True'+'/'+str(self.robot_stop)+'\n'
                            print("[to DB]:",data)
                            db_connect.client_socket.sendall(data.encode('utf-8'))
                            self.last_processed_time_is_coll = collision_time
                        except:
                            print('socket error')
    
     
    def is_dirty(self):
        """
        Robot arm이 동작중이 아닐때 확인(호출)
        object detection이 실행중이여야 하고(거기서 frame정보(웹캠화면)를 받아옴)
        self.table_is_clean 갱신
        """
        frame = self.frame
        is_clean = False
        try:
            result = self.cls_model(frame)
            result_props_per_type_list=result[0].probs.top5conf.tolist()
            if (result[0].probs.top1 == 2) &(result_props_per_type_list[0]>=0.9):
                self.table_is_clean=False
            else:
                is_clean=True
            try:
                data = 'cleanliness/'+ str(datetime.datetime.now())+'/'+is_clean+'\n'
                print("[to DB]:",data)
                db_connect.client_socket.sendall(data.encode('utf-8'))
            except:
                print('socket error')
        except:
            print("The cleanliness of the table cannot be verified at this time.")
            is_clean=False # 일반적으로는 깨끗할 것으로 가정
        
        self.table_is_clean = is_clean

        return is_clean
                    
###############################################################################################
class Cam_Front_Object_Detection(): # 정면을 바라보는 웹캠
    def __init__(self):
        super().__init__()
        # YOLOv8 모델 경로 설정
        self.model_weight_path_receive_check = './cam_front_1115_best.pt'  # receive_check에 사용할 모델 경로
        self.model_weight_path_customer_check = './yolov8n-face.pt'  # customer_check에 사용할 모델 경로

        # 모델 로드
        self.model_receive_check = YOLO(self.model_weight_path_receive_check, verbose=False)
        self.model_customer_check = YOLO(self.model_weight_path_customer_check, verbose=False)

        self.time_slot = 0.1        # Object detection을 몇초마다 진행할 것인지.
        self.cam_front_flag = False # 종료 신호 관리
        self.results = None


    @staticmethod
    def calculate_distance(center1, center2):
        """두 점 사이의 유클리드 거리 계산"""
        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5


    # def object_detection(self, mode=""):
    def object_detection(self):
        cap = cv2.VideoCapture(4)  # 웹캠 열기

        if not cap.isOpened():
            print("Failed to open the webcam.")
            return

        while True:
            if self.cam_front_flag:  # 종료 신호가 오면 루프 종료
                break

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # if mode == "receive_check":
            # receive_check 예측 수행
            results = self.model_receive_check(frame, conf=0.25, verbose=False)
            self.results_receive = results
            for result in self.results_receive:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if self.model_receive_check.names[class_id] == 'robot_gripper':
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    elif self.model_receive_check.names[class_id] == 'icecream_cup':
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # elif mode == "customer_check":
            # customer_check 예측 수행
            results = self.model_customer_check(frame, conf=0.25, verbose=False)
            self.results_customer = results
            for result in self.results_customer:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    if self.model_customer_check.names[class_id] == 'face':
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.imshow("Cam Front", frame)
            cv2.imshow("Cam Table", cam_table_object_detection.predicted_frame)

            cv2.waitKey(1)
            time.sleep(self.time_slot)

        self.cam_front_flag = False
        cap.release()
        cv2.destroyAllWindows()


    def receive_check(self):
        if self.results_receive is None:
            print("No results to process.")
            return 'stop'

        receive_state = 'n'  # 초기 상태는 'n'
        time_limit = 10.0  # 15초 동안 확인
        total_time = 0.0 
        gripper_position = None
        cup_position = None

        distance_threshold = 70  # icecream_cup과 robot_gripper 간의 최소 거리 (픽셀)
        cup_lost_time = 0.0  # icecream_cup이 사라진 시간을 추적

        while total_time < time_limit:  # 15초 동안 확인 (100ms마다 체크)
            # 결과를 self.results로부터 처리
            gripper_found = False
            cup_found = False

            for result in self.results_receive:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if self.model_receive_check.names[class_id] == 'robot_gripper':  # robot_gripper 감지
                        gripper_position = ((x1 + x2) // 2, (y1 + y2) // 2)  # 그리퍼 중앙 좌표 계산
                        gripper_found = True

                    elif self.model_receive_check.names[class_id] == 'icecream_cup':  # icecream_cup 감지
                        cup_position = ((x1 + x2) // 2, (y1 + y2) // 2)  # 컵 중앙 좌표 계산
                        cup_found = True

            if cup_found == True:
                distance = self.calculate_distance(gripper_position, cup_position)
                # 두 객체의 거리가 일정 이상 멀어진 경우
                if distance > distance_threshold:
                    cup_lost_time += self.time_slot  # 3초 동안 멀어진 상태를 유지했는지 확인
                    print(f"Distance between gripper and cup: {distance} (Threshold: {distance_threshold})")
                else:
                    cup_lost_time = 0  # 두 객체가 가까워지면 상태 초기화
            else:
                cup_lost_time += self.time_slot

            # 3초 이상 멀어졌으면 'y' 전송
            if cup_lost_time >= 3.0:  # 3초 이상 (100ms 기준으로 30)
                receive_state = 'y'
                print("Ice cream received, sending 'y'.")
                break

            # 경과 시간 증가
            total_time += self.time_slot
            time.sleep(self.time_slot)  # 100ms마다 상태를 체크

        # 시간 제한을 초과하면 'n' 전송
        if total_time >= time_limit and receive_state == 'n':
            print(f"Time limit of {time_limit} seconds reached, no ice cream received. Sending 'n'.")
            receive_state = 'n'

        self.results_receive = None

        return receive_state

    def customer_check(self):
        if self.results_customer is None:
            print("No results to process.")
            return "n"

        customer_position = None
        max_face_size = 0
        max_face_position = None

        min_face_size = 5000

        time_limit = 10.0  # 10초 동안 확인
        total_time = 0.0  # 전체 경과 시간 추적
        detection_start_time = None  # 얼굴 감지 시작 시간

        while total_time < time_limit:
            for result in self.results_customer:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    if self.model_customer_check.names[class_id] == 'face':
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_size = box_width * box_height

                        if box_size >= min_face_size:
                            if box_size > max_face_size:
                                max_face_size = box_size
                                max_face_position = (x1, y1, x2, y2)
            
            if max_face_position:
                if detection_start_time is None:  # 첫 번째 얼굴 감지 시 시작 시간 기록
                    detection_start_time = 0.0
                detection_start_time += self.time_slot

                # 3초 동안 얼굴이 계속 감지되는지 확인
                if detection_start_time >= 3.:  # 3초가 지났다면
                    customer_position = (max_face_position[0] + max_face_position[2]) // 2, (max_face_position[1] + max_face_position[3]) // 2
                    print(f"Customer detected at {customer_position}. Sending 'y' with coordinates.")
                    return f"y/{customer_position[0]},{customer_position[1]}"
            else:
                detection_start_time = None  # 얼굴이 감지되지 않으면 시작 시간을 초기화
                print("No customer detected.")

            max_face_size = 0.0
            max_face_position = None
            total_time += self.time_slot  # 경과 시간 증가
            time.sleep(self.time_slot)
                

        # 10초가 지나면 'n'을 전송
        if total_time >= time_limit:
            print(f"Time limit of {time_limit} seconds reached, No customer detected. Sending 'n'.")
            return "n"
        
        self.results_customer = None

        return "n"
###############################################################################################

if __name__ == "__main__":
    cam_table_object_detection = Cam_Table_Object_Detection()
    # cam_table_thread = threading.Thread(target=cam_table_object_detection.object_detection)
    # cam_table_thread.start()

    cam_front_object_detection = Cam_Front_Object_Detection()
    # cam_front_thread = threading.Thread(target=cam_front_object_detection.object_detection)
    # cam_front_thread.start()
    # time.sleep(1)
    # cam_front_customer_thread = threading.Thread(target=cam_front_object_detection.object_detection, kwargs={"mode":"customer_check"})
    # cam_front_receive_thread = threading.Thread(target=cam_front_object_detection.object_detection, kwargs={"mode":"receive_check"})
    # cam_front_receive_thread.start()
    # time.sleep(3)
    # print(cam_front_object_detection.receive_check())
    # cam_table_thread.start()
    # cam_front_thread.start()
    # time.sleep(2)
    # cam_front_object_detection.customer_check()

    
    
    robot_connect = Robot_Connect()
    robot_connect_thread = threading.Thread(target=robot_connect.connect_robot)
    robot_connect_thread.start()

    db_connect = DB_Connect()
    db_connect_thread = threading.Thread(target=db_connect.connect_db)
    db_connect_thread.start()
