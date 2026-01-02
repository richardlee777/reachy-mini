sadass## 3주차: 고급 동작 제어

### 학습 목표

- 다양한 보간(interpolation) 방법 활용
- 실시간 동작 제어
- 모터 상태 관리

### 실습 내용

python study/Jaewook/3weeks.py
python -m reachy_mini.daemon.app.main --sim
1. **보간 방법 비교**
    
    ```python
    # linear, minjerk, cartoon, ease 비교
    methods = ["linear", "minjerk", "cartoon", "ease"]
    
    for method in methods:
        print(f"방법: {method}")
        mini.goto_target(
            head=create_head_pose(y=10, mm=True),
            duration=2.0,
            method=method
        )
        time.sleep(1)
        mini.goto_target(head=create_head_pose(), duration=2.0, method=method)
        time.sleep(1)
    
    ```
    1. 보간법(Interpolation)의 차이 이해하기

goto_target의 method 인자를 바꾸면 로봇의 움직임 느낌이 완전히 달라집니다.

Linear: 일정한 속도로 움직입니다. 기계적인 느낌이 강합니다.

Minjerk: 가속과 감속이 부드럽습니다. 로봇 공학에서 가장 많이 쓰이는 자연스러운 동작입니다.

Ease: 시작은 느리고 끝은 빠르게, 혹은 그 반대로 움직여 탄성 있는 느낌을 줍니다.

Cartoon: 애니메이션처럼 약간의 과장된 움직임(오버슈트 등)을 섞어 로봇에게 개성을 부여합니다.



2. **실시간 동작 제어**
    
    ```python
    import time
    import numpy as np
    
    # 사인파 궤적 추종
    with ReachyMini() as mini:
        mini.set_target(head=create_head_pose())
        start = time.time()
    
        while True:
            t = time.time() - start
            if t > 10:
                break
    
            y = 10 * np.sin(2 * np.pi * 0.5 * t)
            mini.set_target(head=create_head_pose(y=y, mm=True))
            time.sleep(0.01)
    
    ```
2. 실시간 제어: set_target 활용
2주차까지 쓴 goto_target은 "2초 동안 저기로 가"라는 예약 명령이었다면, set_target은 **"지금 당장 이 위치로!"**라는 명령입니다. 촘촘한 시간 간격(0.01초)으로 이 명령을 내리면 로봇이 아주 매끄러운 곡선을 그리며 움직입니다.

    
3. **모터 제어**
    
    ```python
    # 모터 활성화
    mini.enable_motors()
    
    # 컴플라이언스 모드 (중력 보상)
    mini.make_motors_compliant()
    
    # 모터 비활성화
    mini.disable_motors()
    
    ```

3. 3주차 실습 및 과제 가이드
실습 시 주의사항: enable_motors 에러 대처
사용하시는 버전(reachy-mini 1.2.5)에 따라 enable_motors() 명령어가 turn_on() 에러 때처럼 존재하지 않을 수 있습니다. 만약 에러가 난다면 해당 줄을 지우고 진행하세요. 시뮬레이션에서는 기본적으로 모터가 활성화된 상태입니다.

과제 1: 원형 궤적 그리기 (힌트)
8자 패턴은 sin(t)와 sin(2t)를 썼지만, **원(Circle)**은 cos(t)와 sin(t)를 사용합니다

Python
# 원형 궤적 힌트 코드 (with문 안에서 사용)
start = time.time()
while time.time() - start < 5: # 5초 동안 회전
    t = time.time() - start
    y = 10 * np.cos(2 * np.pi * 0.5 * t) # 반경 10mm
    z = 10 * np.sin(2 * np.pi * 0.5 * t)
    mini.set_target(head=create_head_pose(y=y, z=z, mm=True))
    time.sleep(0.01)
과제 2: 중력 보상(Compliant) 분석
make_motors_compliant()를 실행하면 로봇의 모터 힘이 풀립니다.

시뮬레이션에서의 동작: 마우스로 로봇 머리를 클릭해서 드래그해 보세요. 힘없이 따라오거나 중력에 의해 아래로 처지는 것을 관찰할 수 있습니다.

분석 포인트: 모터가 켜졌을 때(Stiff)와 꺼졌을 때(Compliant) get_state()로 읽어오는 관절 값의 차이를 확인해 보세요.


4. **안전 범위 테스트**
    
    ```python
    # 안전 범위를 벗어나는 명령 시도
    # 로봇은 가능한 가장 가까운 유효 위치로 이동
    pose = create_head_pose(roll=-50, degrees=True)  # 제한: -40~40도
    mini.goto_target(head=pose)
    
    # 실제 도달한 위치 확인
    current_pose = mini.get_current_head_pose()
    print("현재 머리 포즈:", current_pose)
    
    ```
    

 4. 3주차 통합 실습 코드 (3weeks.py)
에러 가능성을 최소화한 통합 코드입니다. study/Jaewook/3weeks.py를 만들어 실행해 보세요.

Python

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np
import time

with ReachyMini() as mini:
    # [1] 보간법 비교 실습
    for method in ["linear", "minjerk", "cartoon"]:
        print(f"현재 보간 방식: {method}")
        mini.goto_target(head=create_head_pose(y=15, mm=True), duration=1.5, method=method)
        mini.goto_target(head=create_head_pose(y=-15, mm=True), duration=1.5, method=method)
    
    # [2] 실시간 사인파 추종 (5초간)
    print("실시간 사인파 제어를 시작합니다.")
    start = time.time()
    while time.time() - start < 5:
        t = time.time() - start
        y_val = 12 * np.sin(2 * np.pi * 0.5 * t)
        mini.set_target(head=create_head_pose(y=y_val, mm=True))
        time.sleep(0.01)

    print("3주차 기초 실습 완료!")

### 과제

- 원형 궤적을 따라 머리를 움직이는 프로그램 작성
- 중력 보상 모드에서 머리를 움직였을 때의 동작 분석.

