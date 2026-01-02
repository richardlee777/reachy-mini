2주차 실습의 핵심은 로봇을 단순히 '각도'로 움직이는 것을 넘어, 공간 좌표를 활용해 더 생동감 있게 제어하는 것입니다. 올려주신 실습 내용을 바탕으로 핵심 개념과 과제 해결 힌트를 정리해 드립니다.

1. 핵심 함수 이해하기
create_head_pose: 머리의 위치(x, y, z)와 회전(roll, pitch, yaw) 값을 하나의 '자세(pose)' 묶음으로 만들어주는 도구입니다.

mm=True: 단위를 밀리미터로 설정합니다.

degrees=True: 단위를 도(degree)로 설정합니다.

goto_target: "현재 위치에서 목표 위치까지 지정한 시간(duration) 동안 부드럽게 움직여라"라는 명령입니다. 시뮬레이션에서 가장 많이 쓰이는 안전한 명령입니다.

set_target (비교): "지금 당장 이 위치로 가라"는 명령입니다. duration 개념이 없어 로봇이 순간적으로 툭 치듯 움직일 수 있으므로 주의가 필요합니다.

2. 과제 1: 8자 패턴으로 머리 움직이기
8자 패턴은 수학적으로 리사주 도형(Lissajous curve) 원리를 이용하면 쉽습니다. y축(좌우)과 z축(상하)에 서로 다른 주기의 사인(sin) 함수를 적용합니다.

import numpy as np
import time

# 8자 그리기를 위한 시간 파라미터
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np
import time

# 1. 로봇 연결 및 세션 시작
with ReachyMini() as mini:
    print("2주차 제어 및 8자 패턴 실습을 시작합니다!")
    
    # [실습 1] 기본 동작 (생략 가능하지만 확인차 유지)
    mini.goto_target(head=create_head_pose(y=-10, mm=True), duration=1.0)
    time.sleep(0.5)

    # [과제] 8자 패턴 동작 (반드시 with문 내부에 있어야 합니다!)
    print("8자 패턴 시작...")
    for t in np.linspace(0, 2 * np.pi, 25): # 25개 지점으로 정교하게 구성
        y_val = 15 * np.sin(t)      # 좌우 이동
        z_val = 10 * np.sin(2 * t)  # 상하 이동
        
        # 세션이 열려 있는 상태에서 명령 전송
        mini.goto_target(
            head=create_head_pose(y=y_val, z=z_val, mm=True), 
            duration=0.15
        )
        time.sleep(0.05) # 시뮬레이터와 통신 속도 조절

    # [과제] 감정 표현 마무리
    print("놀람 표현 후 종료!")
    mini.goto_target(antennas=np.deg2rad([90, 90]), duration=0.5)
    time.sleep(1)

# 이 지점에서 세션이 안전하게 닫힙니다.
print("모든 동작이 완료되었습니다.")

     
 3. 과제 2: 안테나로 감정 표현하기

안테나는 로봇의 귀와 같아서 각도에 따라 감정이 확연히 달라집니다.

기쁨 (Joy): 양쪽 안테나를 V자 모양으로 높게 세우고 빠르게 까딱거립니다.

antennas=np.deg2rad([45, 45])

슬픔 (Sadness): 양쪽 안테나를 축 늘어뜨립니다.

antennas=np.deg2rad([-30, -30])

놀람 (Surprise): 안테나를 수직으로 빳빳하게 세웁니다.

antennas=np.deg2rad([90, 90])

궁금함 (Curious): 한쪽 안테나만 세우고 머리를 살짝 기울입니다.

antennas=np.deg2rad([45, 0]), roll=10

