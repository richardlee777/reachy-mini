# Week 01: Windows에서 MuJoCo를 활용한 Reachy Mini 시뮬레이션 환경 구축

## 개요

이 문서는 Windows 환경에서 MuJoCo 물리 엔진을 사용하여 Reachy Mini 로봇을 시뮬레이션하는 방법을 다룹니다. Reachy Mini는 Pollen Robotics에서 개발한 오픈소스 데스크탑 휴머노이드 로봇으로, 실제 하드웨어 없이도 MuJoCo 시뮬레이션을 통해 애플리케이션을 프로토타이핑할 수 있습니다.

이 가이드는 **uv**를 사용하여 빠르고 효율적인 개발 환경을 구축하는 방법을 다룹니다.

## 빠른 시작 (uv 사용)

이미 uv가 설치되어 있다면, 다음 명령어로 바로 시작할 수 있습니다:

```bash
# 1. uv 설치 (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 프로젝트 디렉토리 생성
mkdir reachy_mini_project
cd reachy_mini_project

# 3. Python 3.11 환경 생성
uv venv --python 3.11

# 4. 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 5. Reachy Mini + MuJoCo 설치
uv pip install reachy-mini[mujoco]

# 6. 시뮬레이션 실행
reachy-mini-daemon --sim
```

브라우저에서 `http://localhost:8000` 접속하여 시뮬레이션 확인!

## 1. 사전 요구사항

### 시스템 요구사항
- **운영체제**: Windows 10/11 (64-bit)
- **Python**: 3.10 ~ 3.13 (uv가 자동으로 관리)
- **Git**: 최신 버전 (Git LFS 지원 필요)
- **uv**: Python 패키지 관리자 (pip보다 빠름)

### 필수 도구 설치

#### uv 설치

uv는 Rust로 작성된 초고속 Python 패키지 관리자입니다. pip보다 10-100배 빠르며, Python 버전 관리도 함께 제공합니다.

**Windows PowerShell에서 설치:**
```powershell
# PowerShell에서 실행
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**설치 확인:**
```bash
uv --version
```

#### Git LFS 설치
Reachy Mini 리포지토리는 대용량 파일(STL 모델 등)을 관리하기 위해 Git LFS를 사용합니다.

```bash
# Git LFS 설치 확인
git lfs version

# 설치되지 않은 경우
# https://git-lfs.github.com/ 에서 다운로드 후 설치
```

## 2. Reachy Mini SDK 설치

### 방법 1: uv를 사용한 글로벌 설치 (권장)

```bash
# MuJoCo 지원과 함께 설치
uv tool install reachy-mini --with reachy-mini[mujoco]

# 또는 더 간단하게
uv pip install reachy-mini[mujoco]
```

### 방법 2: 프로젝트별 가상환경 설정 (권장)

uv를 사용하면 프로젝트별로 독립적인 환경을 쉽게 만들 수 있습니다.

```bash
# 프로젝트 디렉토리 생성 및 이동
mkdir reachy_mini_project
cd reachy_mini_project

# uv로 가상환경 생성 및 활성화
uv venv

# Windows에서 가상환경 활성화
.venv\Scripts\activate

# reachy-mini 및 MuJoCo 의존성 설치
uv pip install reachy-mini[mujoco]
```

### 방법 3: 소스코드에서 개발 모드로 설치

```bash
# 리포지토리 클론
git clone https://github.com/orocapangyo/reachy_mini.git
cd reachy_mini

# Git LFS 파일 다운로드
git lfs pull

# uv로 가상환경 생성
uv venv

# 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 개발 모드로 설치
uv pip install -e .[mujoco]
```

### uv의 장점

- **속도**: pip보다 10-100배 빠른 패키지 설치
- **Python 버전 관리**: `uv python install 3.11` 명령으로 Python 버전 자동 설치
- **의존성 해결**: 더 정확하고 빠른 의존성 해결
- **캐싱**: 효율적인 패키지 캐싱으로 반복 설치 시간 단축

## 3. MuJoCo 모델 구조 이해하기

### 디렉토리 구조

Reachy Mini 프로젝트의 3D 모델 파일은 다음과 같이 구성되어 있습니다:

```
reachy_mini/
├── src/reachy_mini/
│   ├── descriptions/reachy_mini/
│   │   ├── mjcf/                      # MuJoCo 모델 파일
│   │   │   ├── assets/                # STL 파일 (87개)
│   │   │   │   ├── collision/         # 충돌 메시
│   │   │   │   ├── *.stl              # 3D 메시 파일
│   │   │   │   └── *.blend            # Blender 모델
│   │   │   ├── scenes/                # 시뮬레이션 씬
│   │   │   ├── reachy_mini.xml        # 메인 MuJoCo 모델
│   │   │   ├── scene.xml              # 씬 설정
│   │   │   ├── joints_properties.xml  # 관절 속성
│   │   │   └── config.json            # 설정 파일
│   │   └── urdf/                      # URDF 모델 파일 (ROS 호환)
│   │       ├── assets/                # STL 파일 (173개)
│   │       ├── robot.urdf             # 표준 URDF
│   │       ├── robot_no_collision.urdf
│   │       └── robot_simple_collision.urdf
```

### MuJoCo XML 파일 구조

`reachy_mini.xml` 파일의 주요 구성:

```xml
<mujoco model="reachy_mini">
  <!-- 컴파일러 설정: meshdir="assets"로 STL 파일 경로 지정 -->
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <asset>
    <!-- 43개의 메시 파일 정의 -->
    <mesh name="body_down_3dprint" file="body_down_3dprint.stl"/>
    <mesh name="head_front_3dprint" file="head_front_3dprint.stl"/>
    <!-- ... -->
  </asset>

  <worldbody>
    <!-- 로봇 바디 계층 구조 -->
    <body name="body_foot_3dprint">
      <geom type="mesh" mesh="body_foot_3dprint"/>
      <!-- Stewart 플랫폼, 헤드, 안테나 등 -->
    </body>
  </worldbody>

  <actuator>
    <!-- 9개의 액추에이터: yaw(1) + Stewart(6) + 안테나(2) -->
  </actuator>
</mujoco>
```

### 주요 STL 파일 목록

#### 필수 구조 부품
- `body_down_3dprint.stl` - 본체 하단
- `body_top_3dprint.stl` - 본체 상단
- `body_foot_3dprint.stl` - 베이스
- `head_front_3dprint.stl` - 헤드 전면
- `head_back_3dprint.stl` - 헤드 후면

#### Stewart 플랫폼 부품
- `stewart_main_plate.stl` - 메인 플레이트
- `stewart_link_rod.stl` - 연결 로드
- `dc15_a01_horn_dummy.stl` - 모터 혼

#### 센서 및 전자부품
- `arducam_case.stl` - 카메라 케이스
- `m12_fisheye_lens.stl` - 어안 렌즈
- `antenna.stl` - 안테나
- `5w_speaker.stl` - 스피커

## 4. MuJoCo 시뮬레이션 실행

### 기본 실행

```bash
# 빈 씬으로 시뮬레이션 시작
reachy-mini-daemon --sim

# 또는 명시적으로 빈 씬 지정
reachy-mini-daemon --sim --scene empty
```

### 오브젝트가 있는 씬으로 실행

```bash
# 테이블과 오브젝트가 있는 씬
reachy-mini-daemon --sim --scene minimal
```

### macOS 사용자 참고사항

macOS에서는 MuJoCo의 `mjpython` 인터프리터를 사용해야 합니다:

```bash
mjpython -m reachy_mini.daemon.app.main --sim
```

## 5. 시뮬레이션 확인 및 제어

### 웹 대시보드 접속

시뮬레이션이 실행되면 다음 URL로 접속할 수 있습니다:

```
http://localhost:8000/
```

### REST API 문서

FastAPI 기반의 OpenAPI 문서를 확인할 수 있습니다:

```
http://localhost:8000/docs
```

### Python SDK로 제어하기

```python
from reachy_mini import ReachyMini

# 로봇 인스턴스 생성 (시뮬레이션 또는 실제 로봇)
reachy = ReachyMini()

# 로봇 상태 확인
print(reachy.is_connected())

# 목표 위치로 이동
reachy.goto_target(x=0.3, y=0.0, z=0.2)
```

## 6. STL 파일 커스터마이징

### STL 파일 수정 방법

1. **Blender로 편집**
   - 리포지토리에 포함된 `.blend` 파일 사용
   - `assets/` 폴더의 STL 파일을 Blender로 임포트
   - 수정 후 STL 형식으로 다시 익스포트

2. **FreeCAD 사용**
   - `.part` 파일을 FreeCAD로 열기
   - 파라메트릭 모델링으로 수정
   - STL 익스포트

3. **Fusion 360 또는 SolidWorks**
   - STL 파일 임포트
   - 메시 편집 기능 사용
   - STL 재익스포트

### 커스텀 STL 파일 적용

1. 수정한 STL 파일을 `src/reachy_mini/descriptions/reachy_mini/mjcf/assets/` 폴더에 배치

2. 필요시 `reachy_mini.xml` 파일에서 메시 참조 업데이트:

```xml
<asset>
  <mesh name="custom_part" file="custom_part.stl"/>
</asset>

<worldbody>
  <body name="custom_body">
    <geom type="mesh" mesh="custom_part" rgba="0.8 0.2 0.2 1"/>
  </body>
</worldbody>
```

3. 시뮬레이션 재시작:

```bash
reachy-mini-daemon --sim
```

## 7. 트러블슈팅

### MuJoCo 설치 오류

```bash
# MuJoCo가 제대로 설치되지 않은 경우
uv pip uninstall mujoco
uv pip install mujoco

# 또는 특정 버전 설치
uv pip install mujoco==3.1.0
```

### Git LFS 파일이 다운로드되지 않는 경우

```bash
# LFS 파일 강제 다운로드
git lfs fetch --all
git lfs pull
```

### STL 파일을 찾을 수 없다는 오류

- `meshdir` 경로가 올바른지 확인
- XML 파일의 상대 경로 확인
- 파일명의 대소문자 일치 여부 확인 (Linux/macOS는 대소문자 구분)

### 시뮬레이션이 느린 경우

```xml
<!-- reachy_mini.xml에서 시각화 옵션 조정 -->
<visual>
  <global offwidth="1280" offheight="720"/>
  <quality shadowsize="1024" offsamples="1"/>
</visual>
```

### Python 버전 문제

uv를 사용하면 Python 버전을 쉽게 관리할 수 있습니다:

```bash
# 사용 가능한 Python 버전 확인
uv python list

# 특정 Python 버전 설치 (예: 3.11)
uv python install 3.11

# 프로젝트에 특정 Python 버전 사용
uv venv --python 3.11

# 현재 사용 중인 Python 버전 확인
uv python --version
```

### 의존성 캐시 문제

```bash
# uv 캐시 정리
uv cache clean

# 특정 패키지 캐시만 제거
uv cache clean reachy-mini
```

## 8. 추가 리소스

### 공식 문서
- [Reachy Mini 공식 웹사이트](https://reachymini.net/)
- [Reachy Mini Developer Center](https://reachymini.net/developers.html)
- [MuJoCo 공식 문서](https://mujoco.readthedocs.io/)
- [uv 공식 문서](https://docs.astral.sh/uv/)

### GitHub 리포지토리
- [orocapangyo/reachy_mini](https://github.com/orocapangyo/reachy_mini) - 이번 학습에 사용한 포크
- [pollen-robotics/reachy_mini](https://github.com/pollen-robotics/reachy_mini) - 원본 리포지토리
- [astral-sh/uv](https://github.com/astral-sh/uv) - uv 패키지 관리자

### 관련 문서
- [Reachy 2 시뮬레이션 가이드](https://docs.pollen-robotics.com/developing-with-reachy-2/simulation/simulation-installation/)
- [Pollen Robotics 리소스](https://www.pollen-robotics.com/reachy-2s-online-documentation/)
- [uv vs pip 성능 비교](https://astral.sh/blog/uv)

## 9. 다음 단계

- MuJoCo 시뮬레이션에서 커스텀 동작 프로그래밍
- 강화학습 환경 구축
- ROS 2 통합 (URDF 파일 활용)
- 실제 하드웨어로 코드 이식

---

**작성일**: 2025-12-12
**참고 리포지토리**: https://github.com/orocapangyo/reachy_mini
**라이선스**: Apache 2.0
