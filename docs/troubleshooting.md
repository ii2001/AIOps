# 실제 오류와 해결

## NaN 입력의 validation 오류 응답 직렬화
실제 NaN 요청 검사에서 Pydantic은 입력을 거절했지만 오류 detail.input의 NaN이 Starlette JSONResponse에서 직렬화되지 않았다.
서버 오류: `ValueError: Out of range float values are not JSON compliant`.
해결 전: 예상422 대신 실제500, tests/check_baseline.py의 상태 검사 실패.
해결: 공통 RequestValidationError handler에서 NaN/inf 입력값을 오류 설명 안에서 문자열로 표현한다. 입력을 모델에 전달하거나 정상 값으로 보정하지 않는다.
해결 후: 실제 요청을 다시 실행해 artifacts/api_verification.json과 checks.log에서 422 확인한다.
원본 실패 기록: artifacts/checks_failed.log 및 api_server_failed.log.

## Git 상태
개인 HAIC 및 새 팀 폴더는 Git 저장소가 아니다. 파일 목록 및 원본 CSV SHA-256, 변경 범위 검증으로 대체하며 Git 초기화/commit은 요청하지 않아 수행하지 않았다.

최종 재검사: NaN/inf 모두 실제 HTTP422. 정상168개 입력은 HTTP200/24출력, 167/169개도422. 전체 체크 PASS. 서버는 검사 후 종료했다.
