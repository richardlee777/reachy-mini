# 2주차: 기본 동작 제어 - 상세 가이드

## 학습 목표

이번 주차에서는 Reachy Mini 로봇의 기본적인 동작 제어 방법을 학습합니다:
- 머리(head)와 안테나(antennas)의 기본 동작 제어 방법 익히기
- `goto_target`과 `set_target`의 차이점과 사용 시기 이해
- 로봇의 안전한 동작 범위(safety limits) 파악

---

## 시작하기 전에: 시뮬레이션 환경 준비

이번 주차 실습을 진행하려면 먼저 **[Week 01](week01.md)** 가이드를 참고하여 MuJoCo 시뮬레이션 환경을 구축해야 합니다.

### 빠른 시작 (이미 Week 01을 완료한 경우)

```bash
# 1. 프로젝트 디렉토리로 이동
cd reachy_mini_project

# 2. 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 3. 시뮬레이션 데몬 실행
reachy-mini-daemon --sim

# 4. 브라우저에서 확인
# http://localhost:8000 접속
```

### 처음 시작하는 경우

Week 01 가이드를 따라 다음 단계를 완료하세요:

1. **uv 설치** (PowerShell)
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Python 환경 및 Reachy Mini 설치**
   ```bash
   mkdir reachy_mini_project
   cd reachy_mini_project
   uv venv --python 3.11
   .venv\Scripts\activate
   uv pip install reachy-mini[mujoco]
   ```

3. **시뮬레이션 실행**
   ```bash
   reachy-mini-daemon --sim
   ```

### 실습 환경 확인

시뮬레이션이 정상적으로 실행되면:
- 터미널에 `INFO: Application startup complete` 메시지 표시
- 브라우저에서 `http://localhost:8000` 접속 시 Reachy Mini 대시보드 확인
- `http://localhost:8000/docs`에서 REST API 문서 확인 가능

### 새 터미널에서 Python 스크립트 실행

시뮬레이션 데몬은 계속 실행 상태로 두고, **새 터미널**을 열어서 Python 코드를 실행합니다:

```bash
# 새 터미널 열기
cd reachy_mini_project
.venv\Scripts\activate

# Python 대화형 모드 실행
python

# 또는 Python 스크립트 파일 실행
python my_script.py
```

**중요:**
- 시뮬레이션 데몬(`reachy-mini-daemon --sim`)은 **백그라운드에서 계속 실행**되어야 합니다.
- 제어 코드는 **별도의 터미널**에서 실행합니다.
- 두 터미널 모두 같은 가상환경을 활성화해야 합니다.

### 예제 코드로 빠르게 테스트하기

GitHub 저장소의 예제 파일을 다운로드하여 실행할 수 있습니다:

```bash
# 저장소 클론 (선택사항)
git clone https://github.com/orocapangyo/reachy_mini.git
cd reachy_mini

# 최소 데모 실행
uv run examples/minimal_demo.py

# 시퀀스 예제 실행
uv run examples/sequence.py
```

**예제 파일 설명:**
- `minimal_demo.py` - 머리와 안테나의 기본적인 사인파 움직임
- `sequence.py` - 다양한 머리 회전과 안테나 동작의 시퀀스
- `look_at_image.py` - 카메라 이미지의 특정 지점을 바라보기

---

## 1. 머리 동작 제어 (Head Control)

### 1.1 기본 개념

Reachy Mini의 머리는 3D 공간에서 위치와 회전을 제어할 수 있습니다. `create_head_pose` 함수를 사용하여 목표 포즈(pose)를 생성합니다.

### 1.2 좌표계 이해

- **x축**: 전후 방향 (앞: 양수, 뒤: 음수)
- **y축**: 좌우 방향 (오른쪽: 양수, 왼쪽: 음수)
- **z축**: 상하 방향 (위: 양수, 아래: 음수)

### 1.3 기본 머리 동작 예제

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    # 머리를 왼쪽으로 10mm 이동 (y축 -10mm)
    pose = create_head_pose(y=-10, mm=True)
    mini.goto_target(head=pose, duration=2.0)

    # 초기 위치로 복귀
    pose = create_head_pose()  # 파라미터 없이 호출하면 기본 위치
    mini.goto_target(head=pose, duration=2.0)
```

**주요 파라미터:**
- `y=-10`: y축 방향으로 -10 이동 (왼쪽)
- `mm=True`: 밀리미터 단위 사용 (기본값은 미터)
- `duration=2.0`: 2초 동안 부드럽게 이동

---

## 2. 회전 동작 (Rotation Control)

### 2.1 회전 축 이해

- **Roll**: x축 중심 회전 (좌우로 기울이기)
- **Pitch**: y축 중심 회전 (끄덕이기)
- **Yaw**: z축 중심 회전 (고개 좌우로 돌리기)

### 2.2 회전 예제

```python
# 머리를 위로 10mm 들고(z축) 롤(roll) 15도 회전
pose = create_head_pose(z=10, roll=15, degrees=True, mm=True)
mini.goto_target(head=pose, duration=2.0)
```

**주요 파라미터:**
- `z=10`: z축으로 10mm 상승
- `roll=15`: x축 기준 15도 회전
- `degrees=True`: 각도를 degree 단위로 입력 (기본값은 radian)

### 2.3 다양한 회전 조합

```python
# 고개를 끄덕이는 동작 (pitch)
pose = create_head_pose(pitch=20, degrees=True)
mini.goto_target(head=pose, duration=1.0)

# 고개를 좌우로 돌리는 동작 (yaw)
pose = create_head_pose(yaw=30, degrees=True)
mini.goto_target(head=pose, duration=1.0)

# 복합 회전: 위를 보면서 왼쪽으로 돌리기
pose = create_head_pose(pitch=15, yaw=-25, degrees=True)
mini.goto_target(head=pose, duration=2.0)
```

---

## 3. 안테나 제어 (Antenna Control)

### 3.1 안테나 기본 구조

Reachy Mini는 2개의 안테나를 가지고 있으며, 각각 독립적으로 제어할 수 있습니다.
- 인덱스 0: 왼쪽 안테나
- 인덱스 1: 오른쪽 안테나

### 3.2 기본 안테나 제어

```python
import numpy as np

# 양쪽 안테나를 45도로 이동
mini.goto_target(antennas=np.deg2rad([45, 45]), duration=1.0)

# 초기 위치로 복귀 (0도)
mini.goto_target(antennas=[0, 0], duration=1.0)
```

**참고:**
- 안테나 각도는 radian 단위로 입력 필요
- `np.deg2rad()` 함수로 degree를 radian으로 변환
- `[왼쪽 안테나, 오른쪽 안테나]` 순서

### 3.3 비대칭 안테나 동작

```python
# 왼쪽 60도, 오른쪽 30도
mini.goto_target(antennas=np.deg2rad([60, 30]), duration=1.5)

# 왼쪽만 움직이기 (오른쪽은 0도 유지)
mini.goto_target(antennas=np.deg2rad([45, 0]), duration=1.0)

# 반대 방향으로 움직이기
mini.goto_target(antennas=np.deg2rad([-30, 30]), duration=1.0)
```

---

## 4. 복합 동작 (Combined Movement)

### 4.1 동시 제어

`goto_target` 함수는 여러 부위를 동시에 제어할 수 있습니다.

```python
# 머리, 안테나, 바디를 동시에 움직이기
mini.goto_target(
    head=create_head_pose(y=-10, mm=True),      # 머리를 왼쪽으로
    antennas=np.deg2rad([45, 45]),               # 안테나를 45도로
    body_yaw=np.deg2rad(30),                     # 몸통을 30도 회전
    duration=2.0                                  # 모두 2초 동안 동시 실행
)
```

### 4.2 순차적 동작

```python
with ReachyMini() as mini:
    # 1단계: 몸통 회전
    mini.goto_target(body_yaw=np.deg2rad(45), duration=2.0)

    # 2단계: 머리를 들어올리며 안테나 펼치기
    mini.goto_target(
        head=create_head_pose(z=15, mm=True),
        antennas=np.deg2rad([60, 60]),
        duration=1.5
    )

    # 3단계: 초기 위치로 복귀
    mini.goto_target(
        head=create_head_pose(),
        antennas=[0, 0],
        body_yaw=0,
        duration=2.0
    )
```

---

## 5. goto_target vs set_target

### 5.1 goto_target

```python
# 부드럽게 이동 (duration 동안 보간)
mini.goto_target(head=pose, duration=2.0)
```

**특징:**
- 현재 위치에서 목표 위치까지 부드럽게 이동
- `duration` 파라미터로 이동 시간 지정
- 블로킹 방식: 이동이 완료될 때까지 대기
- 실제 로봇 동작에 권장

### 5.2 set_target

```python
# 즉시 목표 위치 설정 (보간 없음)
mini.set_target(head=pose)
```

**특징:**
- 즉시 목표 위치로 이동 시도
- 부드러운 동작 보장 없음
- 논블로킹 방식: 명령 후 즉시 다음 코드 실행
- 시뮬레이션이나 빠른 반응이 필요한 경우 사용

### 5.3 사용 예시 비교

```python
# goto_target: 부드러운 동작
with ReachyMini() as mini:
    pose1 = create_head_pose(y=-10, mm=True)
    mini.goto_target(head=pose1, duration=2.0)  # 2초간 부드럽게 이동

    pose2 = create_head_pose(y=10, mm=True)
    mini.goto_target(head=pose2, duration=2.0)  # 다시 2초간 이동

# set_target: 즉각 반응
with ReachyMini() as mini:
    pose1 = create_head_pose(y=-10, mm=True)
    mini.set_target(head=pose1)  # 즉시 이동 시작
    time.sleep(0.5)  # 필요시 수동으로 대기

    pose2 = create_head_pose(y=10, mm=True)
    mini.set_target(head=pose2)  # 즉시 다음 목표로
```

---

## 6. 로봇의 안전 범위 (Safety Limits)

### 6.1 물리적 제약사항

Reachy Mini는 각 관절과 부위에 물리적 한계가 있습니다:

**머리(Head):**
- x, y, z 이동 범위: 제조사 스펙 참조 (일반적으로 ±50mm 정도)
- 회전 범위: 각 축마다 다름 (일반적으로 ±45도 정도)

**안테나(Antennas):**
- 회전 범위: 일반적으로 0도 ~ 90도 사이
- 음수 각도도 가능하나 제한적

**바디(Body Yaw):**
- 회전 범위: ±60도 정도 (제조사 스펙 확인 필요)

### 6.2 안전한 동작을 위한 권장사항

```python
# 좋은 예: 점진적 이동
mini.goto_target(head=create_head_pose(y=-10, mm=True), duration=2.0)
mini.goto_target(head=create_head_pose(y=-20, mm=True), duration=2.0)

# 나쁜 예: 급격한 이동
mini.goto_target(head=create_head_pose(y=-50, mm=True), duration=0.1)  # 너무 빠름!
```

### 6.3 에러 처리

```python
try:
    # 범위를 벗어난 명령
    pose = create_head_pose(y=-100, mm=True)  # 너무 큰 값
    mini.goto_target(head=pose, duration=2.0)
except Exception as e:
    print(f"에러 발생: {e}")
    # 안전한 위치로 복귀
    mini.goto_target(head=create_head_pose(), duration=2.0)
```

---

## 7. GitHub 저장소 예제 코드 분석

이 섹션에서는 [orocapangyo/reachy_mini](https://github.com/orocapangyo/reachy_mini) 저장소의 실제 예제 코드를 분석합니다.

### 7.1 최소 데모 (minimal_demo.py)

이 예제는 머리와 안테나를 사인파 패턴으로 움직이는 기본 데모입니다.

```python
"""Minimal demo for Reachy Mini."""

import time
import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini(media_backend="no_media") as mini:
    # 초기 위치로 이동
    mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=1.0)

    try:
        while True:
            t = time.time()

            # 안테나를 20도 진폭으로 0.5Hz 주파수로 움직임
            antennas_offset = np.deg2rad(20 * np.sin(2 * np.pi * 0.5 * t))

            # 머리를 pitch(끄덕임) 방향으로 10도 진폭으로 움직임
            pitch = np.deg2rad(10 * np.sin(2 * np.pi * 0.5 * t))

            head_pose = create_head_pose(
                roll=0.0,
                pitch=pitch,
                yaw=0.0,
                degrees=False,  # 이미 radian으로 변환했으므로
                mm=False,
            )
            # set_target 사용 - 실시간으로 계속 업데이트
            mini.set_target(head=head_pose, antennas=(antennas_offset, antennas_offset))
    except KeyboardInterrupt:
        pass
```

**핵심 개념:**
- `media_backend="no_media"`: 카메라/마이크 없이 실행
- `set_target()`: 논블로킹 방식으로 빠른 업데이트
- 사인파 함수로 자연스러운 주기적 움직임 생성
- `2 * np.pi * 0.5 * t`: 0.5Hz 주파수 (2초에 1번 왕복)

### 7.2 시퀀스 예제 (sequence.py 주요 부분)

다양한 머리 동작과 안테나 제어를 순차적으로 실행하는 예제입니다.

#### Yaw 회전 (좌우로 고개 돌리기)

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

# 2초 동안 yaw 축(좌우) 회전
s = time.time()
while time.time() - s < 2.0:
    t = time.time() - t0
    # yaw 축으로 최대 0.7 radian (약 40도) 회전
    euler_rot = np.array([0, 0.0, 0.7 * np.sin(2 * np.pi * 0.5 * t)])
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat  # 회전 행렬 설정
    mini.set_target(head=pose, antennas=[0, 0])
    time.sleep(0.01)
```

#### Pitch 회전 (끄덕이기)

```python
# 2초 동안 pitch 축(상하) 회전
s = time.time()
while time.time() - s < 2.0:
    t = time.time() - t0
    # pitch 축으로 최대 0.3 radian (약 17도) 회전
    euler_rot = np.array([0, 0.3 * np.sin(2 * np.pi * 0.5 * t), 0])
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat
    mini.set_target(head=pose, antennas=[0, 0])
    time.sleep(0.01)
```

#### Roll 회전 (좌우로 기울이기)

```python
# 2초 동안 roll 축 회전
s = time.time()
while time.time() - s < 2.0:
    t = time.time() - t0
    # roll 축으로 최대 0.3 radian 회전
    euler_rot = np.array([0.3 * np.sin(2 * np.pi * 0.5 * t), 0, 0])
    rot_mat = R.from_euler("xyz", euler_rot, degrees=False).as_matrix()
    pose[:3, :3] = rot_mat
    mini.set_target(head=pose, antennas=[0, 0])
    time.sleep(0.01)
```

#### 상하 이동 (Z축 평행이동)

```python
# 2초 동안 상하 움직임
s = time.time()
while time.time() - s < 2.0:
    t = time.time() - t0
    pose = np.eye(4)  # 4x4 단위 행렬로 초기화
    pose[:3, 3][2] += 0.025 * np.sin(2 * np.pi * 0.5 * t)  # z축 평행이동
    mini.set_target(head=pose, antennas=[0, 0])
    time.sleep(0.01)
```

#### 안테나 비대칭 움직임

```python
# 2초 동안 안테나를 반대 방향으로 움직임
s = time.time()
while time.time() - s < 2.0:
    t = time.time() - t0
    antennas = [
        0.5 * np.sin(2 * np.pi * 0.5 * t),   # 왼쪽 안테나
        -0.5 * np.sin(2 * np.pi * 0.5 * t),  # 오른쪽 안테나 (반대 방향)
    ]
    mini.set_target(head=pose, antennas=antennas)
    time.sleep(0.01)
```

#### 원형 움직임 (X-Y 평면)

```python
# 5초 동안 원형 경로로 머리 이동
s = time.time()
while time.time() - s < 5.0:
    t = time.time() - t0
    pose[:3, 3] = [
        0.015 * np.sin(2 * np.pi * 1.0 * t),           # x 좌표
        0.015 * np.sin(2 * np.pi * 1.0 * t + np.pi / 2),  # y 좌표 (90도 위상차)
        0.0,                                             # z 좌표
    ]
    mini.set_target(head=pose, antennas=[0, 0])
    time.sleep(0.01)
```

**핵심 개념:**
- `np.eye(4)`: 4x4 단위 행렬 (위치와 회전을 표현하는 변환 행렬)
- `pose[:3, :3]`: 회전 행렬 부분 (3x3)
- `pose[:3, 3]`: 평행이동 벡터 (x, y, z)
- `scipy.spatial.transform.Rotation`: 오일러 각을 회전 행렬로 변환
- 위상차(`np.pi / 2`)를 이용한 원형 움직임

### 7.3 look_at_image.py - 비전 기반 제어

카메라 이미지에서 클릭한 지점을 로봇이 바라보도록 하는 예제입니다.

```python
"""Demonstrate how to make Reachy Mini look at a point in an image."""

import cv2
from reachy_mini import ReachyMini

def click(event, x, y, flags, param):
    """마우스 클릭 이벤트 처리"""
    if event == cv2.EVENT_LBUTTONDOWN:
        param["just_clicked"] = True
        param["x"] = x
        param["y"] = y

with ReachyMini(media_backend="default") as reachy_mini:
    state = {"x": 0, "y": 0, "just_clicked": False}

    cv2.namedWindow("Reachy Mini Camera")
    cv2.setMouseCallback("Reachy Mini Camera", click, param=state)

    while True:
        frame = reachy_mini.media.get_frame()
        cv2.imshow("Reachy Mini Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if state["just_clicked"]:
            # 이미지 좌표를 로봇의 목표 위치로 변환하여 이동
            reachy_mini.look_at_image(state["x"], state["y"], duration=0.3)
            state["just_clicked"] = False
```

**핵심 개념:**
- `media.get_frame()`: 카메라에서 프레임 가져오기
- `look_at_image(x, y)`: 이미지 좌표를 3D 공간 좌표로 변환하여 머리 방향 조정
- OpenCV를 이용한 실시간 비디오 처리

### 7.4 실습: 저장소 예제 실행하기

```bash
# 1. 저장소 클론
git clone https://github.com/orocapangyo/reachy_mini.git
cd reachy_mini

# 2. Git LFS 파일 다운로드 (STL 모델 파일)
git lfs pull

# 3. 개발 모드로 설치
uv pip install -e .[mujoco]

# 4. 시뮬레이션 데몬 실행 (터미널 1)
reachy-mini-daemon --sim

# 5. 예제 실행 (터미널 2)
uv run examples/minimal_demo.py
# 또는
uv run examples/sequence.py
```

---

## 8. 과제 (Assignments)

### 과제 1: 8자 패턴으로 머리 움직이기

**목표:** 머리를 부드럽게 8자 패턴으로 움직이는 프로그램 작성

**힌트:**
```python
import numpy as np
import time
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    # 8자를 그리기 위한 포인트들 계산
    # 매개변수 방정식 사용:
    # x(t) = a * sin(t)
    # y(t) = b * sin(2t)

    num_points = 20  # 8자를 그리기 위한 포인트 개수
    radius = 15  # mm

    for i in range(num_points):
        t = (2 * np.pi * i) / num_points

        y = radius * np.sin(t)
        z = radius * np.sin(2 * t) / 2

        pose = create_head_pose(y=y, z=z, mm=True)
        mini.goto_target(head=pose, duration=0.5)

    # 초기 위치로 복귀
    mini.goto_target(head=create_head_pose(), duration=2.0)
```

**도전 과제:**
- 8자의 크기를 변경할 수 있도록 파라미터화
- 8자를 그리는 속도 조절 기능 추가
- 역방향으로도 8자 그리기

### 과제 2: 안테나로 감정 표현하기

**목표:** 안테나의 움직임만으로 다양한 감정 표현

**구현 예시:**

```python
import numpy as np
import time
from reachy_mini import ReachyMini

def express_joy(mini):
    """기쁨: 안테나를 빠르게 위아래로"""
    for _ in range(3):
        mini.goto_target(antennas=np.deg2rad([60, 60]), duration=0.3)
        mini.goto_target(antennas=np.deg2rad([30, 30]), duration=0.3)
    mini.goto_target(antennas=[0, 0], duration=1.0)

def express_sadness(mini):
    """슬픔: 안테나를 천천히 내리기"""
    mini.goto_target(antennas=np.deg2rad([10, 10]), duration=3.0)
    time.sleep(2)
    mini.goto_target(antennas=[0, 0], duration=2.0)

def express_surprise(mini):
    """놀람: 안테나를 갑자기 펼치기"""
    mini.goto_target(antennas=[0, 0], duration=0.1)
    time.sleep(0.5)
    mini.goto_target(antennas=np.deg2rad([80, 80]), duration=0.2)
    time.sleep(1)
    mini.goto_target(antennas=[0, 0], duration=1.5)

def express_curiosity(mini):
    """호기심: 안테나를 번갈아 움직이기"""
    for _ in range(2):
        mini.goto_target(antennas=np.deg2rad([45, 0]), duration=0.5)
        mini.goto_target(antennas=np.deg2rad([0, 45]), duration=0.5)
    mini.goto_target(antennas=[0, 0], duration=1.0)

def express_anger(mini):
    """화남: 안테나를 빠르고 강하게 움직이기"""
    for _ in range(4):
        mini.goto_target(antennas=np.deg2rad([70, 70]), duration=0.2)
        mini.goto_target(antennas=np.deg2rad([20, 20]), duration=0.2)
    mini.goto_target(antennas=[0, 0], duration=1.0)

# 사용 예시
with ReachyMini() as mini:
    print("기쁨 표현")
    express_joy(mini)
    time.sleep(1)

    print("슬픔 표현")
    express_sadness(mini)
    time.sleep(1)

    print("놀람 표현")
    express_surprise(mini)
    time.sleep(1)

    print("호기심 표현")
    express_curiosity(mini)
    time.sleep(1)

    print("화남 표현")
    express_anger(mini)
```

**도전 과제:**
- 머리 움직임과 결합하여 더 풍부한 감정 표현
- 추가 감정 구현 (두려움, 자신감, 혼란 등)
- 감정 전환 애니메이션 추가

---

## 9. 추가 학습 자료

### 9.1 디버깅 팁

```python
# 현재 위치 확인
with ReachyMini() as mini:
    current_head_pose = mini.get_current_head_pose()
    print(f"현재 머리 위치: {current_head_pose}")

    current_antennas = mini.get_current_antennas()
    print(f"현재 안테나 각도: {np.rad2deg(current_antennas)} degrees")
```

### 9.2 성능 최적화

```python
# 불필요한 대기 시간 제거
# 나쁜 예:
mini.goto_target(head=pose1, duration=2.0)
time.sleep(2.0)  # goto_target이 이미 블로킹이므로 불필요

# 좋은 예:
mini.goto_target(head=pose1, duration=2.0)  # 자동으로 2초 대기
mini.goto_target(head=pose2, duration=2.0)  # 바로 다음 동작
```

### 9.3 일반적인 오류와 해결방법

| 오류 | 원인 | 해결방법 |
|------|------|----------|
| 로봇이 움직이지 않음 | ReachyMini 연결 실패 | `with ReachyMini() as mini:` 구문 확인 |
| 각도가 이상함 | degree/radian 혼동 | `degrees=True` 또는 `np.deg2rad()` 사용 확인 |
| 급격한 움직임 | duration이 너무 짧음 | duration을 1.0 이상으로 증가 |
| 목표 위치에 도달 못함 | 물리적 한계 초과 | 목표값을 안전 범위 내로 조정 |

---

## 10. 다음 주 예고

3주차에서는 다음 내용을 학습합니다:
- 센서 데이터 읽기 (카메라, 거리 센서 등)
- 환경 인식 및 반응형 동작
- 복잡한 시퀀스 프로그래밍
- 변환 행렬을 이용한 고급 제어

---

## 참고 자료

### 공식 문서
- [Reachy Mini 공식 문서](https://docs.pollen-robotics.com/)
- [NumPy 공식 문서](https://numpy.org/doc/)
- [SciPy 공식 문서](https://docs.scipy.org/doc/scipy/reference/spatial.transform.html)

### GitHub 저장소
- [orocapangyo/reachy_mini](https://github.com/orocapangyo/reachy_mini) - 이번 학습에 사용한 포크
- [pollen-robotics/reachy_mini](https://github.com/pollen-robotics/reachy_mini) - 원본 리포지토리

### 관련 학습 자료
- [Week 01: 시뮬레이션 환경 설정](week01.md)
- [MuJoCo 공식 문서](https://mujoco.readthedocs.io/)
