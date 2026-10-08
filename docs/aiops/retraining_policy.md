# Warm-start 및 독립 Deployment Gate

실제기본시나리오가미탐지되어 자동실행은없다. 별도 `manual-diagnostic-primary-drift-not-detected` 수동진단으로 실제weights/metrics/Gate/승격만검증했다. as-of=2025-12-31, 미래데이터사용없음.

최근56일: 2025-11-06T00:00:00+09:00~2025-12-31T23:00:00+09:00.
앞42일fine-tuning: optimizer fit 첫35일(~2025-12-10T23:00:00+09:00), 내부EarlyStopping 마지막7일(2025-12-11T00:00:00+09:00~2025-12-17T23:00:00+09:00). Gate targets 마지막14일 2025-12-18T00:00:00+09:00~2025-12-31T23:00:00+09:00는fit/ES에 사용하지 않았다. Gate의 첫input168시간은 이미알려진이전context이므로앞42일과input만겹칠수있고label은겹치지않는다.

Production v1 weights를clone하고학습전 np.array_equal로동일성을확인했다. 기존Train-fitted scaler_v2를유지했다. LR1e-4, max20epoch, ES patience4/best weights restore, 실제5epochs, shuffle=False, seed42.

Gate는동일변경환경에서알려진과거168시간을두모델에똑같이제공한다. primary검출실험의고정원본prediction과Gate의현재관측input replay는별도목적이다. MWh복원/0floor/metrics는공용함수재사용.

Production MAE=177.250890, RMSE=286.797654, WAPE=47.362817%. Candidate MAE=174.774538, RMSE=283.277904, WAPE=46.701117%. 절대상한83.696140%와상대개선을모두확인한실제결과=PASS.

PASS만새Registryversion및Productionalias를승격하고캐시를clear한다. FAIL은Run/모델artifact를보존하며기존version을유지한다. 이번실제Run은PASS였고FAIL처리는controlled unit check 및source branch검증이다. 운영중FAIL이실제로발생했다고쓰지않는다. Gate평가기간을다음튜닝의validation으로재사용하면독립성보장이약해지므로반복실험/후속운영은새chronological holdout을확보해야한다.
