# 고정 +15% 구조적 Drift 실험

시나리오: 신규 태양광 설비 편입으로 거래량 규모가 증가. 첫 완전Test forecast일~마지막날의 중간 날짜 **2025-12-08**부터 actual만1.15배로 변경한다. Train peak의1%=23.94099994 MWh보다 큰 관측에만 적용한다. 이는 의미있는발전량의 사전 고정 휴리스틱이며 정확한 일출/일몰 classifier가 아니다. 작은 야간값과0은 그대로 유지한다. 원본CSV/모델/기존예측은 변경하지 않는다.

동일 이후 기간 원본 WAPE=51.128592%, 주입후=49.819659%. 주입전 기간 WAPE=38.863186%는 다른 기간이라 그것과의 단순 비교만으로drift효과를 주장하지 않는다.

첫threshold초과=None,2일연속=None,3일확정=None,delay=None. None은 미발생을 뜻한다. **+15%는 탐지되지 않았고 자동재학습/자동배포는 실행하지 않았다.**

주입은 실제업무상의가정을 적용한 명시적simulation이지 새 관측데이터라고 주장하지 않는다. 정상reference 예측을 고정한실험과 현재Production캐시를 사용하는live API를 구분한다. +20/+25는 sensitivity_analysis.json에 별도로 기록했고 threshold나 primary15%를 바꾸지 않았다.

same-after normal L1 numerator=98787.502867, 주입후=110658.638989; actual denominator=193213.813378→222118.417590. WAPE는 상대비율이므로 actual규모증가가 반드시WAPE증가로 이어지지 않는다. 이를 탐지성공으로 바꾸어기록하지 않는다.
