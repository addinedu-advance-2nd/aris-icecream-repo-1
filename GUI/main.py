# 터미널 설치 파일들
#pip3 install pyqt5  
#sudo apt install python3-pyqt5  
#sudo apt install pyqt5-dev-tools
#sudo apt install qttools5-dev-tools


import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt, QStringListModel, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QPixmap
import socket
import time, datetime
import json
import threading, subprocess
import queue

import glob
import cv2
from ultralytics import YOLO
from pymilvus import MilvusClient
import shutil
import torch
from PIL import Image
from sklearn.preprocessing import normalize
import numpy as np
from facenet_pytorch import InceptionResnetV1
from collections import Counter
from gtts import gTTS

from IPython.display import display

# 파일경로
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "image")

# 이미지 파일 경로
menu1_path = os.path.join(image_dir, "strawberry.png")
menu2_path = os.path.join(image_dir, "vanilla.png")
menu3_path = os.path.join(image_dir, "choco.png") 
Topping1_path = os.path.join(image_dir, "topping1.png")
Topping2_path = os.path.join(image_dir, "topping2.png")
Topping3_path = os.path.join(image_dir, "topping3.png")
Topping0_path = os.path.join(image_dir, "no.png")
position_path = os.path.join(image_dir,"position.png")
soldout_path = os.path.join(image_dir, "sold_out.png") 
payment_path = os.path.join(image_dir, "payment.png")
most_path = os.path.join(image_dir, "most.png")
last_path = os.path.join(image_dir, "last.png")
mostlast_path = os.path.join(image_dir, "mostlast.png")
# UI 경로
main_file_path = os.path.join(base_dir, "main.ui")
topping_file_path = os.path.join(base_dir, "topping.ui")
memberinfo_file_path = os.path.join(base_dir, "memberinfo.ui")
position_file_path = os.path.join(base_dir,'position.ui')
splashscreen_file_path = os.path.join(base_dir,"splashscreen.ui")
manager_pw_file_path = os.path.join(base_dir,"manager_pw.ui")
manager_file_path = os.path.join(base_dir,"manager.ui")
payment_file_path = os.path.join(base_dir,"payment.ui")
memberconfirm_file_path = os.path.join(base_dir,"memberconfirm.ui")


main_page_class = uic.loadUiType(main_file_path)[0]
topping_page_class = uic.loadUiType(topping_file_path)[0]
memberinfo_page_class = uic.loadUiType(memberinfo_file_path)[0]
position_page_class = uic.loadUiType(position_file_path)[0]
splashscreen_page_class = uic.loadUiType(splashscreen_file_path)[0]
manager_pw_page_class = uic.loadUiType(manager_pw_file_path)[0]
manager_page_class = uic.loadUiType(manager_file_path)[0]
payment_page_class = uic.loadUiType(payment_file_path)[0]
memberconfirm_page_class = uic.loadUiType(memberconfirm_file_path)[0]




# 관리자 페이지
class ManagerDialog(QDialog,manager_page_class):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        # 탭 환경설정
        self.topping_sold_out = [False, False, False]
        self.tabWidget.setStyleSheet("""
        QTabBar::tab {
        width: 360px;  /* 각 탭의 너비 */
        height: 50px;  /* 각 탭의 높이 */
        font-size: 14pt;  /* 글꼴 크기 */
        }
        """)
        self.tabWidget.setTabText(0, "상품 관리")
        self.tabWidget.setTabText(1, "연결 상태")
        # 연결 상태 설정
        self.connect_robot.setStyleSheet("background-color: #FF0000;")
        self.connect_db.setStyleSheet("background-color: #FF0000;")
        self.robot_connection_status()
        self.DB_connection_status()

        self.menu_sold_btn_1.clicked.connect(lambda: self.parent().sold_out_image(1))
        self.menu_sold_btn_2.clicked.connect(lambda: self.parent().sold_out_image(2))
        self.menu_sold_btn_3.clicked.connect(lambda: self.parent().sold_out_image(3))
        
        self.menu_sell_btn_1.clicked.connect(lambda: self.parent().image_to_original(1))
        self.menu_sell_btn_2.clicked.connect(lambda: self.parent().image_to_original(2))
        self.menu_sell_btn_3.clicked.connect(lambda: self.parent().image_to_original(3))

        self.top_sold_btn_1.clicked.connect(lambda: self.toggle_sold_out(0))
        self.top_sold_btn_2.clicked.connect(lambda: self.toggle_sold_out(1))
        self.top_sold_btn_3.clicked.connect(lambda: self.toggle_sold_out(2))

        # 토핑 품절 해제 버튼 연결
        self.top_sell_btn_1.clicked.connect(lambda: self.clear_sold_out(0))
        self.top_sell_btn_2.clicked.connect(lambda: self.clear_sold_out(1))
        self.top_sell_btn_3.clicked.connect(lambda: self.clear_sold_out(2))

    def toggle_sold_out(self, topping_index):
        # 해당 토핑의 상태를 반전하고 변경 사항을 부모로 전달
        self.topping_sold_out[topping_index] = not self.topping_sold_out[topping_index]
        self.parent().update_topping_status(topping_index, self.topping_sold_out[topping_index])

    def clear_sold_out(self, topping_index):
        # 해당 토핑의 상태를 해제하고 변경 사항을 부모로 전달
        self.topping_sold_out[topping_index] = False
        self.parent().update_topping_status(topping_index, False)


        # 연결 로봇암,CV,DB 확인
    def robot_connection_status(self):
        if robot_connect.connected:# #FF0000 빨강 # #00FF00 초록
            self.connect_robot.setStyleSheet("background-color: #00FF00;")
        else:
            self.connect_robot.setStyleSheet("background-color: #FF0000;")

    def DB_connection_status(self):
        if db_connect.connected:
            self.connect_db.setStyleSheet("background-color: #00FF00;")
        else:
            self.connect_db.setStyleSheet("background-color: #FF0000;")



# 관리자 페이지 비밀번호
class ManagerpwDialog(QDialog, manager_pw_page_class):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Aris 아이스크림 키오스크")
        self.manager_dialog = ManagerDialog(self)
        self.password = "1234"  # 비밀번호 설정
        self.input_password = ""  # 사용자 입력 저a
        self.setWindowTitle("Password Keypad")

        self.pw_label.setText("")  # 초기 텍스트를 빈 문자열로 설정

        self.pushButton_0.clicked.connect(lambda: self.add_number("0"))
        self.pushButton_1.clicked.connect(lambda: self.add_number("1"))
        self.pushButton_2.clicked.connect(lambda: self.add_number("2"))
        self.pushButton_3.clicked.connect(lambda: self.add_number("3"))
        self.pushButton_4.clicked.connect(lambda: self.add_number("4"))
        self.pushButton_5.clicked.connect(lambda: self.add_number("5"))
        self.pushButton_6.clicked.connect(lambda: self.add_number("6"))
        self.pushButton_7.clicked.connect(lambda: self.add_number("7"))
        self.pushButton_8.clicked.connect(lambda: self.add_number("8"))
        self.pushButton_9.clicked.connect(lambda: self.add_number("9"))
        
        # delete 버튼 연결
        self.pushButton_del.clicked.connect(self.delete_number)
        
        # 확인 버튼 연결
        self.confirm_btn.clicked.connect(self.check_password)

    def add_number(self, number):
        # 숫자 버튼 클릭 시 실행될 함수
        self.input_password += number  # 입력된 숫자를 누적
        self.pw_label.setText("*" * len(self.input_password))  # 레이블에 입력된 비밀번호 길이만큼 * 표시

    def delete_number(self):
        # delete 버튼 클릭 시 실행될 함수
        self.input_password = self.input_password[:-1]  # 마지막 입력 삭제
        self.pw_label.setText("*" * len(self.input_password))  # 레이블 업데이트

    def check_password(self):
        # 비밀번호 확인
        if self.input_password == self.password:
            self.open_new_window()
        else:
            self.show_error_message()

    def show_error_message(self):
        QMessageBox.warning(self, "오류", "비밀번호를 틀렸습니다 \n 다시 입력해주세요.")
        self.input_password = ""
        self.pw_label.setText("")

    def open_new_window(self):
        self.new_window = ManagerDialog(parent = self.parent())  # ManagerDialog 클래스 인스턴스 생성
        self.new_window.setWindowTitle("Manager Screen")
        self.close()
        self.new_window.show()

    def update_topping_status(self, topping_index, sold_out_status):
        # WindowClass로 전달
        self.parent().update_topping_status(topping_index, sold_out_status)

class MemberConfirmDialog(QDialog,memberconfirm_page_class):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.member.clicked.connect(self.open_main_window)
        self.nonmember.clicked.connect(self.show_signup_message)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.camera_connected)
        self.timer.start(1)

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_frame)
        self.update_timer.start(30)

    def camera_connected(self):
            # 예시임
            # cam_kiosk.cam_manager(perpose="is_customer") # 가능한 명령1 (고객인지 확인 - 스플레시 스크린 직후)
            # cam_kiosk.cam_manager(perpose="register",member_name="JH",member_phone="9705") # 가능한 명령2 (고객 등록)
            # thread_kiosk = threading.Thread(target=cam_kiosk.cam_manager, kwargs={"perpose":"register","member_name":"JH","member_phone":"9705"})
            # thread_kiosk.start()
            thread_is_customer = threading.Thread(target=cam_kiosk.cam_manager, kwargs={"perpose":"is_customer"})
            thread_is_customer.start()                   

    def update_frame(self):
        # 캠 화면 띄우기
        if cam_kiosk.pixmap is not None: 
            self.label.setPixmap(cam_kiosk.pixmap)
        # 유저가 인식되었을 경우
        if cam_kiosk.customer_checking == False:
            self.update_timer.stop()
            if cam_kiosk.customer_info != ["Unknown","Unknown"]:
            # if cam_kiosk.customer_info == ["Unknown","Unknown"]:
                # 유저 정보 검색
                db_connect.get_member_info(cam_kiosk.customer_info[0],cam_kiosk.customer_info[1])
                time.sleep(1)
                # 환영 메시지를 띄우며 음성으로 인사
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setText(cam_kiosk.customer_info[0]+" 고객님 안녕하세요")
                msg_box.setWindowTitle("Info")
                msg_box.setStandardButtons(QMessageBox.Ok)

                audio_play(cam_kiosk.customer_info[0]+"고객님 "+str(int(my_window.total_order_count)+1)+"번째 방문이시네요 반가워요", "audio_member_hi")
                msg_box.exec_()

                # 메인에 유저 정보 전달
                my_window.user_name = cam_kiosk.customer_info[0]
                my_window.user_phone = cam_kiosk.customer_info[1]
                my_window.name_label.setText(f"<span> <b>{my_window.user_name}</b> 고객님 </span>")

                # 초기화
                cam_kiosk.customer_info = ["Unknown","Unknown"]
                cam_kiosk.customer_checking = True

                # 메인화면으로 진행
                main_window_position = self.pos()
                self.close()  # 현재 화면을 닫음
                my_window.move(main_window_position)
                my_window.update_menu_images()
                my_window.show()
            else:
                # 초기화
                cam_kiosk.customer_checking = True
                # 회원가입 여부 메시지 출력
                self.show_signup_message()
            

    def show_signup_message(self):
        # QMessageBox 생성
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("회원가입 안내")
        msg_box.setText("회원가입하시겠습니까?")

        # '네' 버튼 추가
        btn_yes = msg_box.addButton("네", QMessageBox.YesRole)
        # '이대로 진행' 버튼 추가
        btn_continue = msg_box.addButton("비회원으로 진행", QMessageBox.NoRole)

        # 메시지 박스 실행
        msg_box.exec_()

        # 클릭된 버튼에 따라 행동 처리
        if msg_box.clickedButton() == btn_yes:
            self.open_member_dialog()

        elif msg_box.clickedButton() == btn_continue:
            audio_play("비회원으로 진행합니다", "audio_no_member_hi")
            time.sleep(2)
            # 비회원인 경우.
            my_window.most_menu = "딸기맛"      # DB에서 가져오든 설정하든.
            my_window.last_menu = ""
            my_window.update_menu_images()
            audio_play("오늘은 "+my_window.most_menu+"이 인기가 많아요", "audio_no_member_recommend")
            self.open_main_window()

    def open_main_window(self):
        main_window_position = self.pos()
        self.close()  # 현재 스플래시 화면을 닫음
        my_window.move(main_window_position)
        my_window.show()

    def open_member_dialog(self):
        member_dialog = MemberInfoDialog(self)
        member_dialog.move(self.pos()) 
        member_dialog.exec_()
        self.close()

class SplashScreen(QDialog,splashscreen_page_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def mousePressEvent(self, event):
        audio_play("안녕하세요 고객님 카메라를 봐주세요", "audio_hi")
        # 마우스를 클릭하면 메인 화면으로 전환git a
        if robot_connect.connected:
            robot_connect.client_socket.send("get_ready".encode('utf-8'))
        time.sleep(1)
        self.open_memberconfirm_window()

    def open_memberconfirm_window(self):
        self.close()
        memberconfirm_dialog = MemberConfirmDialog(self)
        memberconfirm_dialog.exec_()


# topping.ui(토핑 선택창) 설정 
class NewWindow(QDialog, topping_page_class):
    def __init__(self, menu_name, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("토핑 선택")
        # 메인 창으로부터 받아오는 값들
        self.topping_sold_out = [False,False,False]
        self.image_label = self.findChild(QLabel, 'pixMap_clicked')
        self.text_label = self.findChild(QLabel, 'menu_name_label_clicked')
        self.menu_price_label = self.findChild(QLabel, 'menu_price_label')

        self.menu_price_label.setText("2000 원")
        self.text_label.setText(menu_name)
        self.frame.lower()  # 프레임뒤로보내는 UI 코드
        # 토핑 이미지 출력
        self.Topping_1.setPixmap(QPixmap(Topping1_path))
        self.Topping_2.setPixmap(QPixmap(Topping2_path))
        self.Topping_3.setPixmap(QPixmap(Topping3_path))
        self.noOption.setPixmap(QPixmap(Topping0_path))
        # 토핑 설명 text
        self.topping_name_label_0.setText("없음")
        self.topping_name_label_1.setText("코코볼")
        self.topping_name_label_2.setText("죠리퐁")
        self.topping_name_label_3.setText("해바라기씨")

        # 이미지에 대면 HandCursor로 변형
        self.Topping_1.setCursor(Qt.PointingHandCursor)
        self.Topping_2.setCursor(Qt.PointingHandCursor)
        self.Topping_3.setCursor(Qt.PointingHandCursor)
        self.noOption.setCursor(Qt.PointingHandCursor)

        self.selected_toppings = []  # 선택된 토핑 저장
        self.Topping_1.mousePressEvent = lambda event: self.toggle_option(self.Topping_1, "코코볼")
        self.Topping_2.mousePressEvent = lambda event: self.toggle_option(self.Topping_2, "죠리퐁")
        self.Topping_3.mousePressEvent = lambda event: self.toggle_option(self.Topping_3, "해바라기씨")
        self.noOption.mousePressEvent = lambda event: self.select_no_option()

        # 토핑창에서의 취소, 담기 버튼
        self.cancel_btn.clicked.connect(self.close)
        self.order_btn.clicked.connect(self.add_to_order)

    def set_topping_sold_out(self, sold_out_list):
        # WindowClass에서 전달받은 품절 상태 저장
        self.topping_sold_out = sold_out_list
        self.update_topping_images()

    def update_topping_images(self):
        # 품절 상태에 따라 이미지 설정
        if self.topping_sold_out[0]:
            self.Topping_1.setPixmap(QPixmap(soldout_path))
        else:
            self.Topping_1.setPixmap(QPixmap(Topping1_path))

        if self.topping_sold_out[1]:
            self.Topping_2.setPixmap(QPixmap(soldout_path))
        else:
            self.Topping_2.setPixmap(QPixmap(Topping2_path))

        if self.topping_sold_out[2]:
            self.Topping_3.setPixmap(QPixmap(soldout_path))
        else:
            self.Topping_3.setPixmap(QPixmap(Topping3_path))

        # "없음"이 클릭되면 모든 토핑의 선택을 해제하고 "없음"만 빨간 테두리
    def select_no_option(self):
        self.clear_all_borders()
        self.noOption.setStyleSheet("border: 5px solid red;")
        self.selected_option = self.noOption
        self.selected_toppings = ["없음"]
        # "없음" 선택이 있을 경우 제거
    def toggle_option(self, selected_label, topping_name):
        if "없음" in self.selected_toppings:
            self.selected_toppings.remove("없음")
            self.noOption.setStyleSheet("border: none;")

        # 선택된 토핑이 이미 선택된 상태라면 테두리를 해제하고 리스트에서 제거
        if topping_name in self.selected_toppings:
            selected_label.setStyleSheet("border: none;")
            self.selected_toppings.remove(topping_name)
        else:
            # 선택된 토핑에 빨간 테두리 표시하고 리스트에 추가
            selected_label.setStyleSheet("border: 5px solid red;")
            self.selected_toppings.append(topping_name)

    # 모든 토핑의 스타일 초기화
    def clear_all_borders(self):
        for topping in [self.Topping_1, self.Topping_2, self.Topping_3]:
            topping.setStyleSheet("border: none;")

    def add_to_order(self):
        order_string = self.text_label.text()
        # 없음을 빼는 코드
        selected_toppings = [topping for topping in self.selected_toppings if topping != "없음"]
        
        # 선택된 토핑을 문자열로 변환하여 추가
        if selected_toppings:
            order_string += " + " + " + ".join(selected_toppings)

        if self.parent():
            self.parent().update_order_list(order_string)

        self.close()

    def set_image(self, pixmap):
        self.image_label.setPixmap(pixmap)

class PositionLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.circle_x = 60  # 원의 x 좌표
        self.circle_x_1 = 330  # 원의 x 좌표
        self.circle_x_2 = 580  # 원의 x 좌표

        self.circle_y = 480  # 원의 y 좌표
        self.circle_radius = 85  # 원의 반지름

    def paintEvent(self, event):
        super().paintEvent(event)  # QLabel 기본 이미지 그리기

        # QPainter를 사용해 원 그리기
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen(QColor(255, 0, 0))  # 빨간색 테두리
        pen.setWidth(5)  # 테두리 두께 설정
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)  # 내부를 채우지 않음
        
        # 원 그리기
        painter.drawEllipse(self.circle_x, self.circle_y, self.circle_radius * 2, self.circle_radius * 2)
        # painter.drawEllipse(self.circle_x_1, self.circle_y, self.circle_radius * 2, self.circle_radius * 2)
        # painter.drawEllipse(self.circle_x_2, self.circle_y, self.circle_radius * 2, self.circle_radius * 2)

class PositionDialog(QDialog):
    def __init__(self,main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("아이스크림 놓는 위치")
        self.setGeometry(100, 100, 800, 800)  # Set a custom window size
        # Main layout
        layout = QVBoxLayout()
        
        # Image and circles
                # Add an instruction label at the top of the dialog
        self.text_label = QLabel("아이스크림을 여기다 놔주세요", self)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setFixedHeight(30)  # Set fixed height to make the text label smaller
        self.text_label.setStyleSheet("font-size: 20px; color: red; font-weight: bold;")
        layout.addWidget(self.text_label)

        # Set up the image with circles
        self.position = PositionLabel(self)
        pixmap = QPixmap(position_path)
        scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.position.setPixmap(scaled_pixmap)
        self.position.setScaledContents(True)
        layout.addWidget(self.position)

        # 타이머 설정: 5초 후 자동으로 창을 닫고 스플래시 화면으로 돌아가기
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)  # 한 번만 실행되도록 설정
        self.timer.timeout.connect(self.complete_service)  # 타이머 종료 시 complete_service 호출
        self.timer.start(9000)  # 5초 (5000 밀리초) 후 실행

        # Apply the layout to the dialog
        self.setLayout(layout)
        self.position.update()
        self.move(self.main_window.pos())

    def complete_service(self):
        self.accept()  # 위치 안내 창 닫기
        #self.main_window.restart_kiosk()  # 스플래시 화면으로 돌아가기
        my_window.initialize_ui()
        my_window.restart_kiosk()

class MemberInfoDialog(QDialog,memberinfo_page_class):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("회원 정보")
        self.mb_regist_btn.clicked.connect(self.register_member)
        self.cam_text.setText("이름과 번호를 입력하고 등록 버튼을 눌러주세요")

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_frame_register)

    def update_frame_register(self):
        if cam_kiosk.pixmap is not None:
            self.cam_label.setPixmap(cam_kiosk.pixmap)
        if cam_kiosk.customer_checking == False:
            self.update_timer.stop()
            if self.parent():
                my_window.update_member_info(self.name, self.phone)
                time.sleep(1)
                my_window.update_menu_images()
                my_window.name_label.setText(f"<span> <b>{my_window.user_name}</b> 고객님 </span>")
                self.clear_member_data()
            audio_play(cam_kiosk.customer_info[0]+"고객님 "+str(my_window.total_order_count+1)+"번째 방문이시네요 반가워요", "audio_member_hi")
            self.close()


    def register_member(self):
        self.name = self.name_textEdit.toPlainText()
        self.phone = self.phone_textEdit.toPlainText()

        # 메인 윈도우에 전송
        if self.name and self.phone:
            self.cam_text.setText("카메라를 보고 얼굴을 좌우로 천천히 움직여주세요")
            # 회원 등록 시 얼굴 인식 진행
            thread_register = threading.Thread(target=cam_kiosk.cam_manager, kwargs={"perpose":"register","member_name":self.name,"member_phone":self.phone})
            thread_register.start()
            self.update_timer.start(30)
        else:
            QMessageBox.warning(self, "입력 오류", "이름과 전화번호를 모두 입력해주세요.")

    def clear_member_data(self):
        self.name_textEdit.clear()  # 이름 입력창 초기화
        self.phone_textEdit.clear()  # 전화번호 입력창 초기화   

class OrderListDialog(QDialog): # 임시 리스트 출력
    def __init__(self, order_list, name, phone, parent=None):
        super().__init__(parent)
        self.setWindowTitle("주문 내역")
        self.setGeometry(100, 100, 300, 400)

        layout = QVBoxLayout()
       # 이름 , 번호 써질 라벨 임시 생성
        self.name_label = QLabel(f"{name}")
        self.phone_label = QLabel(f"{phone}")

        
        # 주문 목록 추가
        self.order_list_widget = QListWidget()
        self.order_list_widget.addItems(order_list)
        button_layout = QHBoxLayout()
        self.confirm_button = QPushButton("결제하기")
        self.cancel_button = QPushButton("취소")
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.cancel_button)


        layout.addWidget(self.name_label)
        layout.addWidget(self.phone_label)
        layout.addWidget(self.order_list_widget)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.confirm_button.clicked.connect(self.open_payment_dialog)
        self.cancel_button.clicked.connect(self.reject)  # 취소 버튼은 다이얼로그를 닫기만 함

    def open_payment_dialog(self):
        self.close()
        payment_dialog = PaymentDialog(self)
        payment_dialog.exec_()
        self.accept()

class PaymentDialog(QDialog,payment_page_class):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("결제화면")
        self.payment_label.setPixmap(QPixmap(payment_path))
        self.yes_btn.clicked.connect(self.open_position_window)
        
    def open_position_window(self):
        current_orders = my_window.order_model.stringList()
        if current_orders != []:
            db_connect.db_update_order(current_orders)
        robot_connect.icecream_order()
        audio_play("주문 되었습니다. 캡슐의 씰을 제거하고 표시된 홈에 캡슐을 두면 아이스크림을 만들어드릴게요", "audio_icecream_order")
        self.close()
        position_dialog = PositionDialog(self)
        position_dialog.exec_()

# 메인 윈도우 클래스
class WindowClass(QMainWindow, main_page_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


        self.setWindowTitle("Aris 아이스크림 판매 키오스크")
        self.most_last_1.hide()
        self.most_last_2.hide()
        self.most_last_3.hide()

        self.topping_sold_out = [False,False,False]
        self.menu1.setPixmap(QPixmap(menu1_path))
        self.menu2.setPixmap(QPixmap(menu2_path))
        self.menu3.setPixmap(QPixmap(menu3_path))

        self.menu1.setCursor(Qt.PointingHandCursor)
        self.menu2.setCursor(Qt.PointingHandCursor)
        self.menu3.setCursor(Qt.PointingHandCursor)

        self.menu1.mousePressEvent = lambda event: self.open_new_window(menu1_path, "딸기맛")
        self.menu2.mousePressEvent = lambda event: self.open_new_window(menu2_path, "바닐라맛")
        self.menu3.mousePressEvent = lambda event: self.open_new_window(menu3_path, "초코맛")

        self.menu_name_label_1.setText("딸기맛")
        self.menu_name_label_2.setText("바닐라맛")
        self.menu_name_label_3.setText("초코맛")
        self.menu_price_label_1.setText("2000 원")
        self.menu_price_label_2.setText("2000 원")
        self.menu_price_label_3.setText("2000 원")
        ###############################################################
        # 리스트 받아옴
        self.most_menu = ""
        self.last_menu = ""
        self.update_menu_images()
        self.receipt.clicked.connect(self.show_receipt)
        self.manager_btn.clicked.connect(self.open_manager_window)
        self.Warning_btn.clicked.connect(self.Warning_event)
        self.Position_btn.clicked.connect(self.open_position_window)

        # 현재 고객 정보
        self.user_name = ""
        self.user_phone = ""
        self.total_order_count = 0
        self.update_receipt_button_state()

        #UI 초기화
        self.initialize_ui()


    def update_menu_images(self):
        most_menu = self.most_menu
        last_menu = self.last_menu
        # Update images based on most ordered flavor
        if most_menu == last_menu:
            # If both are the same, show a shared image (best_last_path)
            if most_menu == "딸기맛":
                self.most_last_1.show()
                self.most_last_1.setPixmap(QPixmap(mostlast_path))  # Use the shared image for both
            elif most_menu == "바닐라맛":
                self.most_last_2.show()
                self.most_last_2.setPixmap(QPixmap(mostlast_path))  # Use the shared image for both
            elif most_menu == "초코맛":
                self.most_last_3.show()
                self.most_last_3.setPixmap(QPixmap(mostlast_path))  # Use the shared image for both
        else:
            if most_menu == "딸기맛":
                self.most_last_1.show()
                self.most_last_1.setPixmap(QPixmap(most_path))
            elif most_menu == "바닐라맛":
                self.most_last_2.show()
                self.most_last_2.setPixmap(QPixmap(most_path))
            elif most_menu == "초코맛":
                self.most_last_3.show()
                self.most_last_3.setPixmap(QPixmap(most_path))
            # Update images based on last ordered flavor
            if last_menu == "딸기맛":
                self.most_last_1.show()
                self.most_last_1.setPixmap(QPixmap(last_path))
            elif last_menu == "바닐라맛":
                self.most_last_2.show()
                self.most_last_2.setPixmap(QPixmap(last_path))
            elif last_menu == "초코맛":
                self.most_last_3.show()
                self.most_last_3.setPixmap(QPixmap(last_path))

        self.order_model = QStringListModel()
        self.order_list_widget.setModel(self.order_model)
        self.order_model.setStringList([])  # 초기 빈 리스트 설정
        # 삭제 버튼
        self.delete_btn_1.clicked.connect(lambda: self.delete_item_at_index(0))
        self.delete_btn_2.clicked.connect(lambda: self.delete_item_at_index(1))
        self.delete_btn_3.clicked.connect(lambda: self.delete_item_at_index(2))
        self.delete_btn_4.clicked.connect(lambda: self.delete_item_at_index(3))

    def initialize_ui(self):
        #회원 정보 초기화
        self.user_name = "" 
        self.user_phone = ""
        # 주문 리스트 초기화
        self.order_model.setStringList([])  
        
    def restart_kiosk(self):
        self.hide()  # 메인 화면 숨기기
        # 스플래시 화면을 다시 표시
        splash_screen = SplashScreen()  # 스플래시 화면을 새로 생성
        splash_screen.exec_()  # 스플래시 화면을 모달로 실행

    def update_topping_status(self, topping_index, sold_out_status):
        # 특정 토핑의 품절 상태만 업데이트
        self.topping_sold_out[topping_index] = sold_out_status

    def sold_out_image(self, button_num):
        if button_num == 1:
            self.menu1.setPixmap(QPixmap(soldout_path))
        elif button_num == 2:
            self.menu2.setPixmap(QPixmap(soldout_path))
        elif button_num == 3:
            self.menu3.setPixmap(QPixmap(soldout_path))

    def image_to_original(self, button_num):
        # 원래 이미지를 설정하는 함수
        if button_num == 1:
            self.menu1.setPixmap(QPixmap(menu1_path))
        elif button_num == 2:
            self.menu2.setPixmap(QPixmap(menu2_path))
        elif button_num == 3:
            self.menu3.setPixmap(QPixmap(menu3_path))

    def open_manager_window(self):
        manager_dialog = ManagerpwDialog(self)
        manager_dialog.exec_()

    # 제조 과정 중 사람 접근 확인 시 안내 문구 - 임시로 버튼 활용
    def open_position_window(self):
        position_dialog = PositionDialog(self)
        position_dialog.exec_()

    def update_receipt_button_state(self):
        if not self.order_model.stringList():
            self.receipt.setEnabled(False)
            self.receipt.setStyleSheet("background-color: grey;")
        else:#e54f40 빨강
            self.receipt.setEnabled(True)
            self.receipt.setStyleSheet("background-color: #e54f40;")

    def Warning_event(self):
        QMessageBox.warning(self,'경고','\n뒤로 물러나 주세요\n로봇은 5초 후에 재동작 합니다\n')

    def delete_item_at_index(self, index):
        # 현재 주문 리스트 가져오기
        current_orders = self.order_model.stringList()

        # index가 리스트 범위 내인지 확인 후 삭제
        if 0 <= index < len(current_orders):
            del current_orders[index]  # 해당 인덱스의 항목 삭제
            self.order_model.setStringList(current_orders)
        self.update_receipt_button_state()

    def update_member_info(self, name, phone):
        # 받은 이름과 전화번호로 레이블을 업데이트
        self.user_name = name
        self.user_phone = phone
        db_connect.db_new_member(name,phone)

    def open_new_window(self, image_path, menu_name):
        self.new_window = NewWindow(menu_name, self)
        self.new_window.set_topping_sold_out(self.topping_sold_out)
        self.new_window.set_image(QPixmap(image_path))
        self.new_window.exec_()
    

    def update_order_list(self, order_string):
        current_orders = self.order_model.stringList()
        current_orders.append(order_string)
        self.order_model.setStringList(current_orders)
        self.update_receipt_button_state()

    def show_receipt(self): #임시 영수증 출력
        current_orders = self.order_model.stringList()
        robot_connect.order_receipt = self.order_model.stringList()
        receipt_dialog = OrderListDialog(current_orders, self.user_name, self.user_phone, self)
        receipt_dialog.exec_()
        self.order_model.setStringList([])


class Robot_Connect():
    def __init__(self):
        super().__init__()
        self.order_press = False
        self.avaliable_jig_A = True
        self.avaliable_jig_B = True
        self.avaliable_jig_C = True
        self.topping_first = False      # 아이스크림보다 토핑을 먼저 받을지 말지. True : 먼저 받음
        self.topping_time = 6.0         # 총 토핑 받는 시간 == 토핑 량 
        self.spoon_angle = 185.0        # 스푼의 위치 (Angle 기준) 기계 기준 오른쪽 185.0 / 중앙 270.0 / 왼쪽 355.0
        self.order_receipt = ""
        self.order_queue = queue.Queue()
        self.flag = False       # 스레드 종료 플래그
        self.connected = False


    # robot arm과의 socket 연결부
    def socket_connect(self):
        self.ADDR = '127.0.0.1'
        self.PORT = 54321
        
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 추후 시간이 된다면 관리자 페이지를 만들어서 로봇과 통신 연결 / 로봇 테스트 / 설정 변경 등 구현
        while True:
            if self.flag == True:
                break
            print("[Robot] 연결 시도 중 입니다.")
            try:
                self.client_socket.connect((self.ADDR, self.PORT))
                self.connected = True
                break
            except socket.error as e:
                # print(f"[Robot] 연결할 수 없습니다: {e}")
                print("[Robot] 5초 후 연결을 재시도 합니다")
                time.sleep(5.0)

        # 연결을 계속 유지함
        # 데이터를 수신받아서 정보를 저장함. ex) 지그 사용 완료 및 서비스 완료.
        while self.connected:
            if self.flag == True:
                break
            time.sleep(1)
            print("[Robot] 메시지 수신 대기중")
            try:
                # 대기 시간 설정
                self.client_socket.settimeout(10.0)
                # 메시지 수령
                self.recv_msg = self.client_socket.recv(1024).decode('utf-8')

                # 메시지가 비어 있는 경우. 연결이 끊겼으므로 재연결을 위해 예외 처리
                if self.recv_msg == '':
                    print("[Robot] received empty msg")
                    raise Exception("empty msg")
                
                print("[Robot] received msg : " + self.recv_msg)

                self.recv_msg = self.recv_msg.split('/')
                if self.recv_msg[0] == 'icecream_service_finish':
                    print("Jig " + self.recv_msg[1] + " icecream service end")
                    print("Set jig " + self.recv_msg[1] + " avaliable")
                    if self.recv_msg[1] == 'A':
                        self.avaliable_jig_A = True
                    elif self.recv_msg[1] == 'B':
                        self.avaliable_jig_B = True
                    elif self.recv_msg[1] == 'C':
                        self.avaliable_jig_C = True
                if self.recv_msg[0] == 'robot_pause':
                    my_window.Warning_event()
                    print("로봇 긴급 정지 신호 수신")
                if self.recv_msg[0] == 'robot resume':
                    print("로봇 동작 재개 신호 수신")
            
            except socket.timeout:
                print('[Robot] MainException: {}'.format(socket.timeout))
            except Exception as e:
                print('[Robot] MainException: {}'.format(e))
                self.connected = False
                print('connection lost')
                # 재연결 시도
                while True:
                    if self.flag == True:
                        break
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
                            print("[Robot] 서버에 연결할 수 없습니다. 2초 후 재시도합니다.")
                        
                    except Exception as e:
                        print('MainException: {}'.format(e))
                        print('except')
                        # pass
                

    def order_receipt_process(self):
        # 우선 주문이 하나만 들어온다고 가정하고 코드 작성
        topping_orders = self.order_receipt[0].split(' + ')
        
        # topping 정보
        kor_topping_list = ["코코볼", "죠리퐁", "해바라기씨"]
        topping_signal = ""
        
        for i in range(len(kor_topping_list)):
            if kor_topping_list[i] in topping_orders[1:]:
                topping_signal += "1"
            else:
                topping_signal += "0"

        if topping_signal == "000":
            topping_no = True
        else:
            topping_no = False
        topping = topping_signal

        # jig 선택부
        # jig의 가용 상태를 저장해두고 비어있는 jig를 A B C 순으로 사용
        if self.avaliable_jig_A:
            jig = "A"
            self.avaliable_jig_A = False
        elif self.avaliable_jig_B:
            jig = "B"
            self.avaliable_jig_B = False
        elif self.avaliable_jig_C:
            jig = "C"
            self.avaliable_jig_C = False
        else:
            # 사용 가능한 jig가 없는 경우
            print("there is no avaliable jig!")

        
        # 메시지 생성부
        data_icecream = {
            "jig" : jig,
            "topping_first" : self.topping_first,
            "topping_no" : topping_no,
            "topping" : topping,
            "topping_time" : self.topping_time,
            "spoon_angle" : self.spoon_angle,
        }

        return data_icecream
        

    # robot arm에 아이스크림 추출 명령 전달 함수
    def icecream_order(self):
        if self.order_receipt:
            data_icecream = self.order_receipt_process()
        else:
            print("주문 내역이 없습니다.")
            return
        msg_type = "icecream"

        json_data = json.dumps(data_icecream)
        data = msg_type + '/' + json_data

        try:
            self.client_socket.sendall(data.encode())
        except BrokenPipeError:
            print("메시지 전송 실패! 연결되어 있지 않습니다.")
        
                

class DB_Connect():
    def __init__(self):
        super().__init__()
        self.connected = False
        self.flag = False       # 스레드 종료 플래그
        self.membership_num="" 
        self.last_msg_time = ""

    def socket_connect(self):
        self.ADDR = '127.0.0.1'          # DB 주소로 변경
        self.PORT = 65432                   # DB의 Port로 변경
        
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        while True:
            if self.flag == True:
                break
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
            if self.flag == True:
                break
            # time.sleep(1)
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
                        print("DB 데이터 처리 중 에러 발생")
                        print("Error:",msg_list[1:])
                    elif msg_list[0] == 'order_updated':
                        print("[from DB]:order_updated")
                    elif msg_list[0] == 'new_member_added': ########새로 가입된 회원
                        print("[from DB]회원 가입 완료")
                        self.membership_num=msg_list[2]
                    elif msg_list[0] == 'already_registered': #######기존 가입한 회원
                        print("[from DB]기존에 등록된 회원:,",msg_list)
                        # 기존 회원의 경우 회원의 선호 정보를 수신
                        last_order_time = msg_list[1] #마지막 주문시간
                        self.membership_num=msg_list[2] # 등록된 회원번호
                        my_window.total_order_count = msg_list[3] # 전체 구매횟수
                        current_count = msg_list[4] # 적립 횟수
                        my_window.most_menu = msg_list[5] # 최다주문맛 (최다메뉴1개가 아니면 "아직 고민중이에요" 반환)
                        most_menu_count = msg_list[6] # 최다주문맛의 주문횟수
                        my_window.last_menu = msg_list[7] # 마지막 주문 맛
                        last_menu_cont = msg_list[8] # 마지막 주문 맛의 연속주문횟수
                        if my_window.most_menu == "Strawberry": # Strawberry 딸기맛 Banana 바닐라맛 초코맛
                            my_window.most_menu = "딸기맛"
                        elif my_window.most_menu == "Banana":
                            my_window.most_menu = "바닐라맛"
                        elif my_window.most_menu == "Chocolate":
                            my_window.most_menu = "초코맛"
                        if my_window.last_menu == "Strawberry": # Strawberry 딸기맛 Banana 바닐라맛 초코맛
                            my_window.last_menu = "딸기맛"
                        elif my_window.last_menu == "Banana":
                            my_window.last_menu = "바닐라맛"
                        elif my_window.last_menu == "Chocolate":
                            my_window.last_menu = "초코맛"

                    elif msg_list[0] == 'member_info':
                        print("[from DB] 요청한 회원 정보 수신 :", msg_list)
                        last_order_time = msg_list[1] #마지막 주문시간
                        self.membership_num=msg_list[2] # 등록된 회원번호
                        my_window.total_order_count = msg_list[3] # 전체 구매횟수
                        current_count = msg_list[4] # 적립 횟수
                        my_window.most_menu = msg_list[5] # 최다주문맛 (최다메뉴1개가 아니면 "아직 고민중이에요" 반환)
                        most_menu_count = msg_list[6] # 최다주문맛의 주문횟수
                        my_window.last_menu = msg_list[7] # 마지막 주문 맛
                        last_menu_cont = msg_list[8] # 마지막 주문 맛의 연속주문횟수
                        if my_window.most_menu == "Strawberry": # Strawberry 딸기맛 Banana 바닐라맛 초코맛
                            my_window.most_menu = "딸기맛"
                        elif my_window.most_menu == "Banana":
                            my_window.most_menu = "바닐라맛"
                        elif my_window.most_menu == "Chocolate":
                            my_window.most_menu = "초코맛"
                        if my_window.last_menu == "Strawberry": # Strawberry 딸기맛 Banana 바닐라맛 초코맛
                            my_window.last_menu = "딸기맛"
                        elif my_window.last_menu == "Banana":
                            my_window.last_menu = "바닐라맛"
                        elif my_window.last_menu == "Chocolate":
                            my_window.last_menu = "초코맛"

                    elif msg_list[0] == 'cleanliness': ###### 테이블 청결도 응답
                        print("[from DB] 청결도 정보 도착:,",msg_list)
                        if  isinstance(msg_list[2],bool):
                            self.is_table_clean = msg_list[2]
                        else:
                            print("테이블의 청결여부 정보가 bool형태가 아님")

                    elif msg_list[0] == 'test': 
                        print("[from DB]test")
                    else:
                        print("[from DB]사전에 정의되지 않은 메시지 형식")
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
                # 소켓 정리
                self.client_socket.shutdown(socket.SHUT_RDWR)
                self.client_socket.close()
                while True:
                    if self.flag == True:
                        break
                    time.sleep(2)
                    try:
                        # 소켓 설정
                        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                        try:
                            self.client_socket.connect((self.ADDR, self.PORT))
                            self.connected = True
                            break
                        except socket.error as e:
                            print("[DB] 서버에 연결할 수 없습니다. 2초 후 재시도합니다.")
                        
                    except Exception as e:
                        print('[DB] MainException: {}'.format(e))
                        print('[DB] except')


    def get_member_info(self, member_name, member_phone):
        datetime_now = str(datetime.datetime.now())

        try:
            # data = 'member_info/'+datetime_now+'/'+member_name+"/"+member_phone+'\n'
            data = 'member_info/'+datetime_now+'/'+member_name+"/"+member_phone+'\n'
            self.client_socket.sendall(data.encode('utf-8'))
        except:
            print('socket error')


    def db_new_member(self,member_name, member_phone): # 회원가입 요청 보내고 결과(이미가입됨, 새로 가입완료됨) 받는 함수
        # default
        members_number = ""

        datetime_now = str(datetime.datetime.now())
        
        # 이미 가입된 회원인지 확인
        try:
            data = 'sign_up/'+datetime_now+'/'+member_name+"/"+member_phone+'\n'
            self.client_socket.sendall(data.encode('utf-8'))
            # 맴버쉽번호 받기
        except:
            print('socket error')
                
        return members_number 
    

    def db_update_order(self,current_orders): # 주문-결제시 판매기록 요청 보냄, 회원일 경우 회원번호도 함께 전달, 업데이트 성공여부 받는 함수
        # default
        #print("func: db_update_order:",current_orders)
        # DB로 전송, 현재는 첫번째 주문 하나만 보냄
        orders = current_orders[0].split(' + ')
        print(orders)
        flavor= orders[0]

        # messege parsing
        kor_flavor_list = ["딸기맛", "바닐라맛", "초코맛"] ######################## global, later
        eng_flavor_list = ["Strawberry", "Banana", "Chocolate"] ######################## global, later
        for i in range(len(kor_flavor_list)):
            if flavor == kor_flavor_list[i]:
                flavor = eng_flavor_list[i]
        # topping 정보 parsing
        kor_topping_list = ["코코볼", "죠리퐁", "하바리기씨"] ######################## global, later
        topping_signal = ""
        
        for i in range(len(kor_topping_list)):
            if kor_topping_list[i] in orders[1:]:
                topping_signal += "1"
            else:
                topping_signal += "0"       
 
        datetime_now = str(datetime.datetime.now())

        print("[to DB]db_update_order:",'new_order'+'/'+datetime_now+'/'+flavor+'/'+topping_signal+'/'+self.membership_num)

        # 회원 비회원 여부
        if self.membership_num != "":
            try:
                data = 'new_order'+'/'+datetime_now+'/'+flavor+'/'+topping_signal+'/'+self.membership_num+'\n'
                self.client_socket.sendall(data.encode('utf-8'))
            except:
                print('socket error')
        else:
            # 비회원 주문 케이스
            try:
                data = 'new_order'+'/'+datetime_now+'/'+flavor+'/'+topping_signal+'/'+'Null\n'
                self.client_socket.sendall(data.encode('utf-8'))
            except:
                print('socket error')


class Cam_Kiosk():
    def __init__(self, is_reset):
        super().__init__()
        self.kiosk_cam_flag = True
        self.time_slot = 2  # Set detection interval in seconds

        self.temp_folder = "./temp_face"
        os.makedirs(self.temp_folder, exist_ok=True)  # Ensure temp folder exists
        self.registered_folder = "./registered_face"
        os.makedirs(self.registered_folder, exist_ok=True)  # Ensure temp folder exists

        # Extract and recognition part ############
        self.yolo_model = YOLO("./yolov8n-face.pt")
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

        self.num_face_pic = 5 # 회원 등록시 촬영 사진 수
        
        self.milvus_db_path = "./customer_face.db"
        self.client = MilvusClient(uri=self.milvus_db_path)
        if is_reset:
            self.make_milvus_db()
            self.load_registered_member()

        self.customer_info = ["Unknown","Unknown"]

        self.pixmap = None
        self.customer_checking = True
        ############################################

    def cam_manager(self, perpose, member_name="Unknown",member_phone="Unknown"):
        # 초기화
        self.customer_info = ["Unknown","Unknown"]
        # 용도별 분류
        if perpose == "register":
            print("Taking photo - register customer")
            # 촬영
            pics= self.capture_faces(self.num_face_pic,5)
            # 등록
            self.customer_info = [member_name, member_phone]
            self.register_face(pics, member_name,member_phone)
            
        elif perpose == "is_customer":
            print("Taking photo - recognize customer")
            # 촬영
            pics= self.capture_faces(1,countdown=1)
            #인식 및 결과 리턴
            member_name,member_phone=self.query_and_display(pics)
            #print(member_name,member_phone)
            self.customer_info = [member_name, member_phone]

        elif perpose == "test":
            print("perpose : Test")
            self.capture_faces(5)

        else:
            print("Wrong perpose - taking photo")
        
        self.customer_checking = False
        if self.customer_info != ["Unknown","Unknown"]:
            print("customer info:",self.customer_info)
        else:
            print("식별할수없어요. 처음방문이신가요?")
        # 임시 촬영 사진 삭제
        #self.clear_temp_folder()

    def clear_temp_folder(self):
        # temp_folder에 있는 모든 파일을 하나씩 삭제
        try:
            for filename in os.listdir(self.temp_folder):
                file_path = os.path.join(self.temp_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        #print(f"Deleted {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        except:
            print("./temp_face doesn't exist")

    def check_images_in_folder(self, num):
        # 특정 폴더 내의 .jpg, .jpeg, .png 파일들을 찾음
        image_files = glob.glob(os.path.join(self.temp_folder, "*.jpg")) + \
                    glob.glob(os.path.join(self.temp_folder, "*.jpeg")) + \
                    glob.glob(os.path.join(self.temp_folder, "*.png"))
        
        # 이미지 파일의 개수가 num개 이상인지 확인
        if len(image_files) == num:
            #print("폴더에 사진 수:",num)
            sign = True
        else:
            #print("폴더에 사진 수:",num)
            sign = False

        return sign

    def face_file_moving (self, member_name, member_phone):
            # file move (temp => registered)
            for i, filename in enumerate(os.listdir(self.temp_folder), start=1):
                temp_file_path = os.path.join(self.temp_folder, filename)
                
                try:
                    if os.path.isfile(temp_file_path):
                        new_filename = f"{member_name}_{member_phone}_{i}.jpg"
                        
                        dest_file_path = os.path.join(self.registered_folder, new_filename)
                        shutil.move(temp_file_path, dest_file_path)
                        #print(f"Moved {temp_file_path} to {dest_file_path}")
                except Exception as e:
                    print(f"Error moving {temp_file_path}: {e}")

            print(f"Face registration complete (ID: {member_name})")

    def capture_faces (self, num_pic, countdown=5):
        self.clear_temp_folder()
    
        cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미. 다른 웹캠을 사용할 경우 번호를 변경
        name = "test"

        if not cap.isOpened():
            print("Failed to open the webcam(Kiosk cam).")
            cap.release()
            return

        print(f"Ready to capture images!")
        
        # 시작 시간 기록
        start_time = time.time()
        countdown_done = False
        last_capture_time = None

        captured_images = 0
        self.customer_checking = True
        while captured_images < num_pic:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image.")
                break

            # YOLOv8 for face detection
            results = self.yolo_model.predict(frame)

            # YOLO의 출력 결과를 가져옴
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Face bounding box coordinates

            if boxes is not None and len(boxes) > 0:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

                # 면적이 가장 큰 박스를 선택
                largest_box_index = areas.argmax()  # 가장 큰 면적을 가진 박스의 인덱스
                largest_box = boxes[largest_box_index]  # 가장 큰 박스

                # 선택된 박스 좌표
                x1, y1, x2, y2 = largest_box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # 얼굴 영역에 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                
                # 카운트다운이 끝났고, 지정된 텀(self.time_slot) 이후에만 촬영
                current_time = time.time()
                if countdown_done and (last_capture_time is None or current_time - last_capture_time >= self.time_slot):
                    face_img = frame[y1:y2, x1:x2]  # Crop face region

                    # Verify face image is not empty
                    if face_img.size == 0:
                        continue
                    
                    # Resize to 160x160
                    face_img_resized = cv2.resize(face_img, (160, 160))

                    # Save the cropped and resized face
                    img_name = os.path.join(self.temp_folder, f"{name}_{captured_images + 1}.jpg")
                    cv2.imwrite(img_name, face_img_resized)
                    print(f"Saved {img_name}")
                    
                    captured_images += 1
                    last_capture_time = current_time

            # 실시간으로 웹캠 영상 송출
            if not countdown_done:
                elapsed_time = int(time.time() - start_time)
                if elapsed_time < countdown:
                    # 화면에 카운트다운 표시
                    cv2.putText(frame, f"Starting in {countdown - elapsed_time}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    countdown_done = True
                    print("Start capturing images!")

            # BGR(OpenCV) -> RGB(Pillow 또는 PyQt 호환) 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Numpy 배열 -> QImage 변환
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # QImage -> QPixmap으로 변환하여 QLabel에 표시
            pixmap = QPixmap.fromImage(qimg)
    
            self.pixmap = pixmap
            # splash_screen.memberconfirm_dialog.label.setPixmap(pixmap)

            # cv2.imshow('YOLOv8 Real-time Prediction', frame)

            # # 실시간 업데이트를 위해 최소 시간 대기
            # if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 강제 종료
            #     break

        # 모든 작업이 끝나면 창 닫기
        cap.release()
        # cv2.destroyAllWindows()

        return num_pic

    def make_milvus_db (self):
        # Set up a Milvus client
        # Create a collection in quick setup mode
        if self.client.has_collection(collection_name="face_embeddings"):
            self.client.drop_collection(collection_name="face_embeddings")
        self.client.create_collection(
            collection_name="face_embeddings",
            vector_field_name="vector",
            dimension=512,
            auto_id=True,
            enable_dynamic_field=True,
            metric_type="COSINE",
        )
        
    def load_registered_member (self):
        # 기존멤버 등록용
        for dirpath, foldername, filenames in os.walk(self.registered_folder):
            for filename in filenames:
                if filename.endswith(".jpg") | filename.endswith(".png"):
                    filepath = dirpath + "/" + filename
                    image_embedding = self.extract_face_embedding(filepath)
                    filename = filename.split('_')
                    print(filename)
                    member_phone = filename[1] ####################################
                    member_name = filename[0] #########################################
                    self.client.insert(
                        "face_embeddings",
                        {"vector": image_embedding, "name":member_name, "phone":member_phone ,"filename": filepath},
                    )

    def register_face (self, pics ,member_name, member_phone):
        # temp폴더에 사진 5개 있을 경우 진행
        is_enough_picture = self.check_images_in_folder(pics)
        if is_enough_picture == False:
            print("no picture to register")
            return
        print(member_phone, member_name)
        i = 1
        for dirpath, _, filenames in os.walk(self.temp_folder):
            for filename in filenames:
                if filename.endswith(".jpg") | filename.endswith(".png"):
                    filepath = dirpath + "/" + filename
                    image_embedding = self.extract_face_embedding(filepath)
                    #########################################
                    final_path = os.path.join(self.registered_folder,member_name+"_"+member_phone+"_"+str(i)+".jpg")
                    self.client.insert(
                        "face_embeddings",
                        {"vector": image_embedding, "name":member_name, "phone":member_phone ,"filename": final_path},
                    )
                    i +=1
        
        # 임시폴더에서 이동
        self.face_file_moving(member_name, member_phone)
        self.clear_temp_folder()

        # 메시지 전송?   dest_file_path
        return
    
    def extract_face_embedding(self, image_paths):
        embeddings = []
        #print("image_paths: ",image_paths)
        detected_face = self.crop_face(image_paths)

        face_tensor = self.preprocess_face(detected_face)
        embedding = self.facenet_model(face_tensor).detach().cpu().numpy()
        embeddings.append(embedding)
        
        if embeddings:
            mean_embedding = np.mean(embeddings, axis=0)
            normalized_embedding = normalize(mean_embedding.reshape(1, -1), norm="l2").flatten()
            return normalized_embedding
        else:
            print("Error - nomalize embedding")
            return "error"
        
    def preprocess_face(self,face_img):
        face_img = cv2.resize(face_img, (160, 160))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img / 255.0
        face_img = np.transpose(face_img, (2, 0, 1))
        face_tensor = torch.tensor(face_img).float().unsqueeze(0)
        return face_tensor
    
    def crop_face(self, image_paths):
        detected_face =[]
        
        face_img = cv2.imread(image_paths)
        if face_img is None:
            print(f"Warning: Unable to load image at {image_paths}. Skipping.")
            
        # YOLOv8로 얼굴 감지
        results = self.yolo_model(source=face_img)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # 검출된 얼굴 바운딩 박스 좌표

        if len(boxes) == 0:
            print(f"Warning: No face detected in {image_paths}. Skipping.")
        else: 
            x1, y1, x2, y2 = map(int, boxes[0])
            detected_face = face_img[y1:y2, x1:x2]

            if detected_face.size == 0:
                print(f"Warning: Detected face is empty in {image_paths}. Skipping.")
                    
        return detected_face

    def query_and_display(self, pics, is_display=False):
        is_enough_picture = self.check_images_in_folder(pics)
        print(is_enough_picture)
        if is_enough_picture == False:
            print("no picture to register")
            return
        
        image_files = [os.path.join(self.temp_folder, f) for f in os.listdir(self.temp_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

        images = []
        name, phone = "Unknown", "Unknown"  # 기본 값 설정

        maybe = []

        for query_image in image_files:

            results = self.client.search(
                "face_embeddings",
                data=[self.extract_face_embedding(query_image)],
                output_fields=["name", "phone", "filename"],
                search_params={"metric_type": "COSINE"},
            )
      
            for result in results:
                for hit in result[:10]:
                    #print("hit:",hit) # 검색결과 ###########################################################################
                    filename = hit["entity"]["filename"]
                    if is_display == True:
                        img = Image.open(filename)
                        img = img.resize((150, 150))
                        images.append(img)

            most = result[0]

            if most["distance"] >= 0.6: 
                name = most["entity"]["name"].split('.')[0]
                name = name.split('_')[0]
                phone = most["entity"]["phone"]
                score = most["distance"]
            
                maybe.append([name,phone,score])

        #print("maybe:",maybe)
        # 가장 많이 등장한 [name, phone] 쌍과 그 빈도 반환
        if maybe != []:
            counter = Counter(tuple(pair) for pair in maybe)
            most_common = counter.most_common(1)
            pair, count = most_common[0]

            name = pair[0]
            phone = pair[1]
            score = pair[2]
            if score >= 0.7: ###########거의 확정
                print("당신의 이름은:",name)
                    
            elif score >= 0.6: ######### 애매한 경우
                print("혹시 ",name,"님 인가요?")
        else:
            print("매칭되는 기록이 없어요(0.6미만)")
            name = "Unknown"
            phone = "Unknown"          

        if is_display == True:
            ## 테스트 목적 디스플레이
            width = 150 * 5
            height = 150 * 2
            concatenated_image = Image.new("RGB", (width, height))

            for idx, img in enumerate(images):
                x = idx % 5
                y = idx // 5
                concatenated_image.paste(img, (x * 150, y * 150))

            display("query")
            display(Image.open(query_image).resize((150, 150)))
            display("results")
            display(concatenated_image)

        return (name, phone)


# 음성 출력 함수
def audio_play(audio_text="", audio_file_name="temp"):
    file_path = "./audio_file/" + audio_file_name + ".mp3"
    
    # 음성 파일 생성
    language = 'ko'
    myobj = gTTS(text=audio_text, lang=language, slow=False)
    myobj.save(file_path)

    try:
        subprocess.Popen(["mplayer", '-quiet', file_path])
    except Exception as e:
        print("Exception : "+e)
    # tts = os.system("mplayer "+audio_file_name+".mp3")

            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    my_window = WindowClass()
    # 스플래시 화면을 보이기 위한 함수 정의
    
    robot_connect = Robot_Connect()
    robot_connect_thread = threading.Thread(target=robot_connect.socket_connect)
    robot_connect_thread.start()
    print("robot_connect thread start")

    db_connect = DB_Connect()
    db_connect_thread = threading.Thread(target=db_connect.socket_connect)
    db_connect_thread.start()
    print("db_connect_thread start")
    
    cam_kiosk = Cam_Kiosk(is_reset=False) # 새로하거나 이미지기준 멤버 등록하려면 True

    # Cam_kiosk.cam_manager(perpose="is_customer") # 가능한 명령1 (고객인지 확인 - 스플레시 스크린 직후)

    # Cam_kiosk.cam_manager(perpose="register",member_name="hj",member_phone="0000") # 가능한 명령2 (고객 등록)

    # 초기 스플래시 화면 표시
    splash_screen = SplashScreen()
    splash_screen.exec_()
    # 스플래시 화면이 닫힌 후 메인 윈도우 표시
    my_window.show()
    app.exec_()
    
    # 프로그램 종료 시 스레드 종료
    robot_connect.flag = True
    db_connect.flag = True
    sys.exit()