from reachy_mini import ReachyMini
import time

try:
    # 로봇과 연결 시도
    with ReachyMini() as mini:
        print("====================================")
        print("축하합니다! 로봇 연결에 최종 성공했습니다.")
        print("====================================")
        
        # 현재 로봇의 관절 상태를 출력합니다.
        state = mini.get_state()
        print(f"로봇 현재 관절 상태: {state}")
        
except Exception as e:
    print(f"연결에 실패했습니다. 에러 내용: {e}")