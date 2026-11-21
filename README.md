# NexaSemi FAB-M16 | NexaPortal Enterprise & NexaAI Agent

가상 반도체 제조 기업 **넥사세미(NexaSemi)**의 1c 16Gb DDR5 BCAT Recess Etch 공정에서 발생한 Data Retention 불량을 AI 에이전트(Function Calling + RAG + DOE/ANOVA)로 규명·해결한 프로젝트입니다.

## 사내 인트라넷 접속 정보 (GitHub Pages)
- **테스트 계정 (사번 / 이메일)**: `NX260715` 또는 `cr.lee@nexasemi.com`
- **비밀번호**: `NexaFab16!` (로그인 화면의 `테스트 계정 자동 입력` 버튼 지원)

## 파일 구성
- `index.html`: NexaSSO 로그인 게이트 및 6대 사내 업무 모듈을 통합한 가상 기업 인트라넷 (`NexaPortal Enterprise`)
- `nexa_ai_agent.py`: Gemini API + Function Calling 기반 멀티스텝 공정 트러블슈팅 AI 에이전트
- `data/01_cp_retention_test_data.csv`: CP Data Retention Test & CD 계측 데이터 (60 Wafers)
- `data/02_fdc_etch_sensor_log.csv`: Recess Etch 설비 FDC 센서 시계열 & PM 누적 웨이퍼 로그 (60 Wafers)
- `data/03_doe_stepdown_recipe_validation.csv`: Over-Etch Step-Down 레시피 최적화 27-Run 3요인 3수준 DOE 데이터
- `requirements.txt`: 필수 Python 패키지 목록

## Python AI 에이전트 실행 방법

```bash
pip install -r requirements.txt

# 1. API 키 없이 로컬 통계/검증 파이프라인 즉시 실행
python nexa_ai_agent.py --offline

# 2. Gemini API Function Calling 실시간 실행 (PowerShell 기준)
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
python nexa_ai_agent.py
```
