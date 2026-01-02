import numpy as np
import time
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

def draw_circle(mini, num_cycles=2, radius=20, speed=0.1, z_offset=0, plane='xy'):
    """
    원형 궤적으로 머리를 움직이는 함수
    
    Args:
        mini: ReachyMini 인스턴스
        num_cycles: 원을 몇 번 그릴지 (기본값: 2)
        radius: 원의 반지름 (밀리미터, 기본값: 20mm)
        speed: 각 포즈 간 이동 속도 (초, 기본값: 0.1초)
        z_offset: Z축 오프셋 (밀리미터, 기본값: 0)
        plane: 원을 그릴 평면 ('xy', 'xz', 'yz', 기본값: 'xy')
    """
    # 원형 궤적을 그리기 위한 점의 개수
    num_points = 50
    
    for cycle in range(num_cycles):
        print(f"원형 궤적 {cycle + 1}/{num_cycles} 시작...")
        
        for i in range(num_points):
            # 파라미터 t: 0부터 2π까지
            t = 2 * np.pi * i / num_points
            
            # 원형 패턴의 파라미터 방정식
            if plane == 'xy':
                # XY 평면 (수평 원)
                x = radius * np.cos(t)
                y = radius * np.sin(t)
                z = z_offset
            elif plane == 'xz':
                # XZ 평면 (수직 원, 앞뒤-위아래)
                x = radius * np.cos(t)
                y = 0
                z = radius * np.sin(t) + z_offset
            elif plane == 'yz':
                # YZ 평면 (수직 원, 좌우-위아래)
                x = 0
                y = radius * np.cos(t)
                z = radius * np.sin(t) + z_offset
            else:
                raise ValueError("plane은 'xy', 'xz', 'yz' 중 하나여야 합니다.")
            
            # 머리 포즈 생성 (밀리미터 단위)
            pose = create_head_pose(x=x, y=y, z=z, mm=True)
            
            # 목표 위치로 이동
            mini.goto_target(head=pose, duration=speed)
        
        print(f"원형 궤적 {cycle + 1}/{num_cycles} 완료")
    
    # 초기 위치로 복귀
    print("초기 위치로 복귀 중...")
    initial_pose = create_head_pose()
    mini.goto_target(head=initial_pose, duration=1.0)
    print("완료!")

# 실행 예시: XY 평면에서 원 그리기 (수평 원)
with ReachyMini() as mini:
    draw_circle(
        mini, 
        num_cycles=2,      # 2번 반복
        radius=20,         # 20mm 반지름
        speed=0.1,         # 각 포즈마다 0.1초
        z_offset=0,        # 높이 오프셋 없음
        plane='xy'         # XY 평면
    )