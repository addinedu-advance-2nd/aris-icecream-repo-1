import sqlite3
import datetime, time
from snowflake import SnowflakeGenerator
import os
import shutil
import pprint

import threading
import socket
import json

"""
------------------------------------
    TCP통신을 위한 DB
------------------------------------
"""
class Database():
    def __init__(self, **kwargs):
        self.kiosk_photo_path='../kioskphoto/' # 회원 얼굴사진 찍은 직후 저장위치, 임시, 기능x
        self.member_photo_path='./face_photo/' # 회원 등록 완료 후 사진 보관 위치, 임시, 기능x
        self.reward_counts=7 # 임시, 기능x
        self.Flavor_list=["Strawberry", "Banana","Chocolate"] # 선택가능한 맛 list, 없는 맛 선택시 오류

        self.Order_db_abspath = "./DB_list/Orders.db"
        self.Membership_db_abspath = "./DB_list/Membership.db"
        self.Reward_preference_db_abs_path = "./DB_list/Reward_preference.db"
        self.Robot_incident_db_abspath = "./DB_list/Robot_incident.db"
        
        self.recv_msg = ""
        self.last_msg_time=""
        # 사진관련? test
        self.new_photo = False

    def socket_connect(self):
        # socket 연결  (연결 될때까지 지속적 시도.)
        self.HOST = '127.0.0.1' 
        self.PORT = 65432
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
                
                print("[GUI] 서버와 연결 시도 중 입니다.")
                time.sleep(1)
                while True:
                    try:
                        self.clientSocket, addr_info = self.serverSocket.accept()
                        print("[GUI] 서버와 연결 성공")
                        break
                    except socket.timeout:
                        print("socket timeout")
                        print("연결을 재시도 합니다")
                break
            except:
                pass

        print("--client info--")
        print(self.clientSocket)

        self.connected = True

        # 지속적으로 메시지 수령.
        while self.connected:
            msg_list = []
            print("[GUI] 메시지 수신 대기 중")
            time.sleep(0.5)
            
            try:
                # 대기 시간 설정
                self.clientSocket.settimeout(10.0)
                # 메시지 수령
                self.recv_msg = self.clientSocket.recv(self.BUFSIZE).decode('utf-8')
                print('received msg : ' + self.recv_msg)
                
                # 메시지가 비어 있는 경우. 연결이 끊겼으므로 재연결을 위해 예외 처리
                if self.recv_msg == '':
                        print('received empty msg')
                        raise Exception('empty msg')
                
                #########################################
                # 메시지는 '/'로 내용 구분되어 오므로, 전처리
                msg_list = self.recv_msg.split('\n')[-2]
                msg_list = msg_list.split('/')
            
            
                # 수신 메시지 처리, 중복실행 방지를 위해 타임스템프 비교를 활용함 
                if self.last_msg_time != msg_list[1]:
                    if msg_list[0] in ['test', 'new_order','sign_up', 'member_info']:
                        # 받은 메시지 타입 확인
                        print(f'message type : {msg_list[0]}')
                        
                        # 세부 메시지별 처리
                        if msg_list[0] == 'new_order': # 주문 내역 저장
                            order_time = msg_list[1]
                            flavor = msg_list[2]
                            topping_signal=msg_list[3]
                            member_n=msg_list[4]
                            print('[from GUI] 새로운 주문(new_order)수신:',order_time,flavor,topping_signal,member_n)

                            # 함수 실행
                            try:
                                self.db_update_order(order_time, flavor,topping_signal,member_n)
                                print('새로운 주문(new_order) - 처리 완료')
                            except:
                                print('Error - order process')

                        elif msg_list[0] == 'sign_up':  # 회원가입
                            sign_time = msg_list[1]
                            sign_name = msg_list[2]
                            sign_phone = msg_list[3]
                            print('[from GUI] 회원가입요청(sign_up)수신:',sign_time,sign_name,sign_phone)

                            # 함수 실행
                            try:
                                self.db_new_member(sign_time,sign_name, sign_phone) 
                                print('회원가입요청(sign_up) - 처리 완료')
                            except:
                                print('Error - sign up process')

                        elif msg_list[0] == "member_info":
                            print("[from GUI] first_face_recognition")
                            member_name = msg_list[2]
                            member_phone = msg_list[3]
                            self.db_first_face_recognition(member_name,member_phone)

                        elif msg_list[0] == "is_table_clean":
                            print("[from GUI] table clean?")
                            data = "is_table_clean/"+str(datetime.datetime.now())+'\n'
                            AI_main.client_socket.sendall(data)
                        elif msg_list[0] == 'test': # 테스트
                            """
                            키오스크쪽에서 test 신호를 보냈다. 
                            """
                            print('[from GUI] test 수신:',msg_list)
                            
                    else:
                        #self.clientSocket.send('ERROR : wrong msg received'.encode('utf-8'))
                        print('got unexpected msg! - 약속된 형식이 아닌 메시지 수신')
                else:
                    #print("이전과 같은 메시지")
                    data='test/0\n'
                    self.clientSocket.sendall((data).encode('utf-8'))
                
                self.last_msg_time = msg_list[1]
                
                #########################################

            except socket.timeout:
                pprint.pprint('[GUI]MainException: {}'.format(socket.timeout))

            except Exception as e:
                pprint.pprint('[GUI]MainException: {}'.format(e))
                self.connected = False
                print('[GUI]connection lost')
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

                        self.serverSocket.bind(self.ADDR)
                        print("[GUI]bind")

                        while True:
                            self.serverSocket.listen(1)
                            print(f'[GUI]reconnecting...')
                            try:
                                self.clientSocket, addr_info = self.serverSocket.accept()
                                self.connected = True
                                break

                            except socket.timeout:
                                print('[GUI]socket.timeout')

                            except:
                                pass
                        break
                    except Exception as e:
                        pprint.pprint('[GUI]MainException: {}'.format(e))
                        print('[GUI]except')
                        # pass

    def db_first_face_recognition (self, member_name, member_phone):

        members_number = ""
        timestamp = str(datetime.datetime.now())
        
        db_sign_ismember,is_mem_flag, members_number = self.is_member(member_name,member_phone)

        if members_number != "": # 성공케이스
            db_sign, member_info = self.greeting_member(membership_n=members_number)
            data='member_info/'+member_info+'\n'
            print("[to GUI]:",data)
            try:
                self.clientSocket.sendall(data.encode('utf-8'))
            except:
                print("socket error - db_first_face_recognition")
        else:
            print("db error - is member에서 반환하는 members_num이 비어있음")
            try:
                data='db_error/'+timestamp+'/greeting_member'+'\n'
                self.clientSocket.sendall(data.encode('utf-8')) # db 멤버쉽정보 접근과정에서 오류
            except:
                print('socket error')
    
    def db_new_member(self,sign_time,sign_name, sign_phone) :
        # default
        is_mem_flag = False
        members_number = ""
        db_sign = False

        datetime_now = sign_time
        
        db_sign_ismember,is_mem_flag, members_number = self.is_member(sign_name,sign_phone)
        
        timestamp = str(datetime.datetime.now())

        if db_sign_ismember & is_mem_flag : # db 통신성공 & 이미 회원 가입이 되어있는 경우 (가입한것 까먹고 또 가입)
            # 회원 적립정보 가져오기
            try:
                db_sign, member_info=self.greeting_member(members_number)
            except:
                print("Error - greeting member")
            
            #print("already signed member, your info: ",member_info)

            # 신호 보내기
            if db_sign == True:
                try:
                    data='already_registered/'+member_info+'\n'
                    print("[to GUI]:",data)
                    self.clientSocket.sendall(data.encode('utf-8'))
                except:
                    print('socket error')
            else:
                try:
                    data='db_error/'+timestamp+'/greeting_member'+'\n'
                    self.clientSocket.sendall(data.encode('utf-8')) # db 멤버쉽정보 접근과정에서 오류
                except:
                    print('socket error')

        elif db_sign_ismember: # db 통신성공 & 신규
            try:
                db_sign, members_number = self.add_new_member(datetime_now, name=sign_name, phone=sign_phone)
                print("new member, your membership number is ",members_number)
            except:
                db_sign = False
                print("Error - add_new_member")
            
            # 신호 보내기
            if db_sign == True:
                try:
                    self.clientSocket.send(('new_member_added/'+timestamp+'/'+str(members_number)+'\n').encode('utf-8'))
                except:
                    print('socket error')
            else:
                try:
                    self.clientSocket.send(('db_error/'+timestamp+'/add_member'+'\n').encode('utf-8')) # db 회원추가 과정에서 오류
                except:
                    print('socket error')

        else: # db 회원여부 확인 과정에서 오류
            db_sign = False
            # 신호 보내기
            try:
                self.clientSocket.send(('db_error/'+timestamp+'/db_new_member'+'\n').encode('utf-8')) 
            except:
                print('socket error')
    
    def db_update_order(self,order_time, flavor,topping_signal,member_n):
        # default 
        datetime_now = order_time

        # 회원 비회원 여부
        try:
            if member_n != "Null":
                db_sign = self.update_order_history(datetime_now=datetime_now, flavor=flavor , toppings=topping_signal,membership_n=member_n)
            else:
                # 비회원 주문update_order_history 케이스
                db_sign = self.update_order_history(datetime_now=datetime_now, flavor=flavor , toppings=topping_signal,membership_n="Null")

            if db_sign :
                self.clientSocket.send(('order_updated/'+order_time+'\n').encode('utf-8'))
            else:
                self.clientSocket.send(('db_error/'+order_time+'/update_order_history-268'+'\n').encode('utf-8'))
        except:
            self.clientSocket.send(('db_error/'+order_time+'/update_order_history-270'+'\n').encode('utf-8'))
            print('socket error')

    # 테이블 생성 -이미 존재할 경우 종료
    def init_database (self): 
        #db_list = ["Orders", "Membership", "Reward_preference"]

        #create table - Order
        con_Order = sqlite3.connect(self.Order_db_abspath)
        cur_Order = con_Order.cursor()
        try:
            cur_Order.execute('''CREATE TABLE Orders(dateTime datetime, flavor text, 
                        toppings text, membership_num integer )''')
        except:
            print("Table [Orders] already exist")

        #create table - Membership
        con_Membership = sqlite3.connect(self.Membership_db_abspath)
        cur_Membership = con_Membership.cursor()
        try:
            cur_Membership.execute('''CREATE TABLE Membership(dateTime datetime, name text, 
                        phone_num text, membership_num integer )''')
        except:
            print("Table [Membership] already exist")

        #create table - Reward_preference
        con_Reward_preference = sqlite3.connect(self.Reward_preference_db_abs_path)
        cur_Reward_preference = con_Reward_preference.cursor()
        try:
            cur_Reward_preference.execute('''CREATE TABLE Reward_preference(
                        update_datetime datetime, membership_num integer, 
                        total_order_count integer, current_count integer, 
                        menu_history text, latest_menu text, latest_menu_cont integer)''')
        except:
            print("Table [Reward_preference] already exist")    


        #create table 
        con_Robot_incident = sqlite3.connect(self.Robot_incident_db_abspath)
        cur_Robot_incident = con_Robot_incident.cursor()
        try:
            cur_Robot_incident.execute('''CREATE TABLE Robot_incident(incident_id integer, incident_time datetime, 
                                       occlusion_level real, collision_with_human text , 
                                       emergency_stop_activated text )''')
        except:
            print("Table [Robot_incident] already exist")

        con_Order.close()
        con_Membership.close()
        con_Reward_preference.close()
        con_Robot_incident.close()
        
    # 주문 이력 추가
    def update_order_history(self,datetime_now, flavor, toppings, membership_n):
        db_sign = ""
        try:
            con_Order = sqlite3.connect(self.Order_db_abspath)
            cur_Order = con_Order.cursor()
            datetime_now = str(datetime_now)
            sql=(datetime_now,flavor,toppings,membership_n)
            # Insert a row of data
            cur_Order.execute("""INSERT INTO Orders VALUES (?,?,?,?)""",sql)

            # save chainge
            con_Order.commit()
      
            # if membership number exist, update member_prefer_rewards DB
            if membership_n != "":
                self.update_member_prefer_rewards(datetime_now,flavor,membership_n)
            
            # end
            con_Order.close()
            db_sign = True
         
        except:
            db_sign = False


        return db_sign

    # 신규회원 - 회원정보&회원적립 및 선호 DB 내역 입력
    def add_new_member(self,datetime_now,name, phone):
        #default
        global member_photo_path
        member_name = ""
        member_phone = ""
        members_number="0"
        db_sign_add=""
        db_sign_coupon=""
        db_sign = "" 

        # 회원 번호 생성
        gen = SnowflakeGenerator(24)
        members_number = next(gen)

        # 회원번호 중복확인
        #check_member_num_duplication(members_number)

        # 이름 처리
        if name:
            member_name = str(name)
        
        # 폰 번호 처리 "01000000000"으로 들어온다고 가정
        if phone:
            member_phone = str(phone)
        
        # 회원 추가
        try:
            # Membership DB - 회원 정보 기록
            con_Membership = sqlite3.connect(self.Membership_db_abspath)
            cur_Membership = con_Membership.cursor()

            sql = (datetime_now,member_name,member_phone,members_number)
            #print(sql)
            # Insert a row of data
            cur_Membership.execute("""INSERT INTO Membership VALUES (?,?,?,?)""",sql)
            # save chainge
            con_Membership.commit()
            # end
            con_Membership.close()

            db_sign_add = True

        except:
            db_sign_add = False
            print("Error - add_new_member / 회원 추가")
            

        # Membership preference & reward DB - 멤버쉽 쿠폰 생성
        try:
           
            #Reward_preference
            con_Reward_preference = sqlite3.connect(self.Reward_preference_db_abs_path)
            cur_Reward_preference = con_Reward_preference.cursor()
            # default value
            total_order=0
            current_count = 0
            menu_history = ""
            for i in range(len(self.Flavor_list)):  
                menu_history += "0_" 
            last_menu="Null"
            last_menu_cont=0

            cur_Reward_preference.execute( """INSERT INTO Reward_preference Values (?,?,?,?,?,?,?)""",
                                            (datetime_now,members_number,total_order,current_count,menu_history,last_menu,last_menu_cont))
            con_Reward_preference.commit()
            con_Reward_preference.close()

            db_sign_coupon = True

        except:
            db_sign_coupon = False
            print("Error - add_new_member / 멤버쉽 쿠폰 생성")

        if db_sign_add & db_sign_coupon: # 두 과정 다 문제 없을 때,, db sign에 True를 보내어 이상없음을 전달
            db_sign = True
        else:
            db_sign = False

        return db_sign, members_number

    # 회원번호 중복확인 - 미구현
    def check_member_num_duplication(self,members_number):
        # 필요시 구축, 회원번호 중복여부 확인용
        is_duplication = False

        return is_duplication

    # 회원번호 조회 - (얼굴인식과 연계예정)
    def is_member(self,member_name,member_phone): 
        db_sign = False
        is_mem_flag=False
        members_number =""
        join_time =""

        # 해당 회원번호가 멤버 목록에 있는지
        # Membership DB - 회원 정보 기록
        con_Membership = sqlite3.connect(self.Membership_db_abspath)
        cur_Membership = con_Membership.cursor()

        sql = (member_name,member_phone)

        try:
            cur_Membership.execute("""SELECT * FROM Membership WHERE name=(?) and phone_num = (?)""",sql)
                
            for row in cur_Membership:
                # 회원 정보 리턴
                join_time=str(row[0])
                print("이미 가입한 이력이 있어요.",join_time)
                members_number = row[3]
                is_mem_flag = True
                
            if join_time == "":
                is_mem_flag = False
            
            db_sign = True
        except:
            print("Error - is_member 정보 가져오기")

        con_Membership.close()
        
        return db_sign,is_mem_flag, members_number

    # 리워드 사용관련 - 미구현
    def is_reward_available(self,membership_n):
        #리워드 사용할때인가?
        global reward_counts
        return

    # 회원의 구매 및 선호기록 
    def update_member_prefer_rewards(self,date, flavor, membership_n):
        
        total_order=0
        current_count = 0
        menu_history=""
        new_menu_history=""
        last_menu=""
        last_menu_cont=0

        #Reward_preference
        con_Reward_preference = sqlite3.connect(self.Reward_preference_db_abs_path)
        cur_Reward_preference = con_Reward_preference.cursor()
        # 주문기록 추가
        cur_Reward_preference.execute("""SELECT * FROM Reward_preference WHERE membership_num=(?)""",(membership_n,))
        last_order=date
        for row in cur_Reward_preference:
            total_order=row[2]+1
            current_count = row[3]+1

            menu_history=row[4]
            flavor_num=self.Flavor_list.index(flavor) 
            menu_list=menu_history.split('_')
            
            for i in range(len(self.Flavor_list)):  
                    if i == flavor_num:
                            new_menu_history += str(int(menu_list[i])+1)+"_" 
                    else:
                            new_menu_history += str(menu_list[i])+"_" 

            # 최근 주문 메뉴가 현재 주문 메뉴와 같으면 연속일수 +1
            last_menu=flavor
            if flavor == row[5]: 
                last_menu_cont = row[6] +1
            else:
                last_menu_cont=1 
        
        cur_Reward_preference.execute( """UPDATE Reward_preference SET update_datetime = (?), total_order_count = (?), 
                                        current_count=(?), menu_history=(?), 
                                        latest_menu=(?), 
                                        latest_menu_cont=(?) WHERE membership_num=(?)""",
                                        (last_order,total_order,current_count,new_menu_history,last_menu,last_menu_cont,membership_n))
        con_Reward_preference.commit()
        
        con_Reward_preference.close()

    # 회원 인식 성공 했을 때 인삿말 등 활용 목적 - 회원 적립상태 반환
    def greeting_member(self,membership_n):
        db_sign = False
        
        last_order="Null"
        total_order=0
        current_count = 0
        menu_history="Null"
        most_menu="Null"
        most_menu_count=0
        last_menu="Null"
        last_menu_cont=0
        
        #Reward_preference
            #Reward_preference
        con_Reward_preference = sqlite3.connect(self.Reward_preference_db_abs_path)
        cur_Reward_preference = con_Reward_preference.cursor()
        # 주문기록 추가
        membership_n = int(membership_n)
        cur_Reward_preference.execute("""SELECT * FROM Reward_preference WHERE membership_num=(?)""",(membership_n,))
            
        try:
            for row in cur_Reward_preference:
                last_order=str(row[0])
                total_order=str(row[2])
                current_count = str(row[3])
                menu_history=str(row[4])
                last_menu=str(row[5])
                last_menu_cont=str(row[6])
                menu_list=menu_history.split('_')
                most_menu_count = str(max(menu_list))
                # 최다주문 맛이 1개로 결정할수없는경우 확인목적
                """dup_cnt = menu_list.count(most_menu_count)
                if dup_cnt >=2 :
                    most_menu = "아직 고민중이에요"
                else: most_menu = self.Flavor_list[menu_list.index(max(menu_list))]
                """
                most_menu = self.Flavor_list[menu_list.index(max(menu_list))]
            db_sign = True
        except:
            print("Error - greeting_member")
            

        # end
        con_Reward_preference.close()

        # 마지막구매일 / 멤버십번호 /전체구매횟수 / 현재적립수 /최다구매메뉴/최다구매메뉴횟수 /마지막주문메뉴 /마지막주문메뉴연속횟수
        member_info = last_order+'/'+str(membership_n)+'/'+ total_order+'/'+current_count+'/'+most_menu+'/'+most_menu_count+'/'+ last_menu+'/'+last_menu_cont

        return db_sign, member_info


class AI_Connect():
    def __init__(self):
        super().__init__()
        self.flag = False       # 스레드 종료 플래그
        self.connected = False

        self.last_msg_time = ""
        self.Robot_incident_db_abspath = "./DB_list/Robot_incident.db"

    def socket_connect(self):
        self.HOST = '127.0.0.1' 
        self.PORT = 23456
        self.ADDR = (self.HOST, self.PORT)

        # 소켓 설정
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.settimeout(10)
        
        self.server_socket.bind(self.ADDR)
        
        while True:

            try:
                # 연결 대기
                self.server_socket.listen(1)
                self.client_socket, addr_info = self.server_socket.accept()
                self.connected = True
                print("[AI] 서버와 연결되었습니다.")
                break
            except  socket.timeout:
                print("[AI] socket timeout")
                print("[AI] 연결을 재시도 합니다.")

        msg_list = []
        while self.connected:
            if self.flag == True:
                break
            time.sleep(1)
            print("[AI] 메시지 수신 대기중")
            try:
                # 대기 시간 설정
                self.client_socket.settimeout(10.0)
                # 메시지 수령
                self.recv_msg = self.client_socket.recv(1024).decode('utf-8')

                # 메시지가 비어 있는 경우. 연결이 끊겼으므로 재연결을 위해 예외 처리
                if self.recv_msg == '':
                    print("[AI] received empty msg")
                    raise Exception("empty msg")
                
                print("[AI] receieved msg : "+ self.recv_msg)

                #########################################
                # 수신 메시지 처리, 중복실행 방지를 위해 타임스템프 비교를 활용함 
                # 전달받는 메시지 처리부 ["incident/일시/사고강도_iou값/사람과충돌?/정지여부/기록"]
                
                msg_list = self.recv_msg.split('\n')[-2]
                msg_list = msg_list.split('/')
               
                if self.last_msg_time != msg_list[1]:
                  
                    if msg_list[0] == 'incident':
                        print('[from AI] 사고정보(incident)수신:',msg_list)
                        self.incident_record(msg_list)
                    elif msg_list[0]== 'cleanliness':
                        print("[from AI] table청결정보 수신",msg_list)
                        data = "cleanliness/"+str(datetime.datetime.now())+'/'+msg_list[-1]
                        DB_main.clientSocket.sendall(data)
                    elif msg_list[0] == 'test': #테스트 신호
                        print('[from AI] test:',msg_list)
                        #self.client_socket.send('test'.encode('utf-8'))
                    else:
                        print("[from AI] 사전에 정의되지 않은 메시지 수신:",msg_list)
                    
                else:
                    print("[AI] 이미 처리한 메시지")

                self.last_msg_time = msg_list[1]
            
                #########################################
            
            except socket.timeout:
                print('[AI] MainException: {}'.format(socket.timeout))
            except Exception as e:
                print('[AI] MainException: {}'.format(e))
                self.connected = False
                print('[AI] 서버 connection lost')
                # 재연결 시도
                while True:
                    time.sleep(1)
                    try:
                        # 소켓 정리
                        self.server_socket.shutdown(socket.SHUT_RDWR)
                        self.server_socket.close()
                        
                        # 소켓 설정
                        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        self.server_socket.settimeout(10)

                        self.server_socket.bind(self.ADDR)
        
                        # 연결 대기
                        while True:
                            if self.flag == True:
                                break
                            try:
                                self.server_socket.listen(1)
                                self.client_socket, addr_info = self.server_socket.accept()
                                self.connected = True
                                print("[AI] 서버와 연결되었습니다.")
                                break
                            except  socket.timeout:
                                print("[AI] socket timeout")
                                print("[AI] 연결을 재시도 합니다.")
                        
                    except Exception as e:
                        print('[AI] MainException: {}'.format(e))
                        print('[AI] except')
                        # pass

    def incident_record(self, msg_list):
        db_sign = False
        db_sign_get_last = False

        # 사고 내역 기록
        timestamp = str(datetime.datetime.now())
        try:
            con_Order = sqlite3.connect(self.Robot_incident_db_abspath)
            cur_Order = con_Order.cursor()

            # 사고기록 id 생성
            try:
                db_sign_get_last, incident_id=self.get_incident_last_index()
            except:
                db_sign_get_last = False
                incident_id = 9998

            incident_id = int(incident_id)+1

            incident_time = msg_list[1]
            occlusion_level = float(msg_list[2])
            collision_with_human = msg_list[3]
            emergency_stop_activated = msg_list[4]
            
            sql=(incident_id,incident_time,occlusion_level,collision_with_human,emergency_stop_activated)

            # Insert a row of data
            cur_Order.execute("""INSERT INTO Robot_incident VALUES (?,?,?,?,?)""",sql)

            # save chainge
            con_Order.commit()
            
            # end
            con_Order.close()
            db_sign = True
        except:
            db_sign = False

        # 메시지 보내기
        if (db_sign_get_last == False) | (db_sign == False): # 마지막 인덱스 가져오기 실패 케이스
            print("Error-get_incident_last_index")
        else:
            try:
                data = 'incident_saved/'+timestamp+'\n'
                self.client_socket.sendall((data).encode('utf-8'))
                print("성공적으로 사고기록 저장 완료:",timestamp)
            except:
                print("[AI]socket error")
        
    
    def get_incident_last_index(self):
        db_sign = False
        last_index = 0

        try:
            con_Incident = sqlite3.connect(self.Robot_incident_db_abspath)
            cur_Incident = con_Incident.cursor()

            cur_Incident.execute("""SELECT incident_id FROM Robot_incident ORDER BY incident_time DESC LIMIT 1 """) # 최근이 위에 오도록
            
            for row in cur_Incident:
                last_index = int(row[0])
            db_sign = True
        except:
            print("Error - get_incident_last_index")
        
        return db_sign, last_index


if __name__ == '__main__':
    DB_main = Database()
    # 소켓 통신용 스레드 동작. (지속적으로 메시지를 수령해야하기에.)
    DB_main.init_database()
    socket_thread = threading.Thread(target=DB_main.socket_connect)
    socket_thread.start()
    print('socket_thread start')    

    AI_main = AI_Connect()
    ai_socket_thread = threading.Thread(target=AI_main.socket_connect)
    ai_socket_thread.start()
    print('ai_socket_thread start')    

    

