"""Charts and documentation from actual, frozen primary experiment and explicitly separate diagnostic."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,hashlib,difflib
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.metrics import evaluate
from src.aiops_registry import OUT,current,client
from src.config import ROOT
from scripts.run_aiops import simulate


def main():
 policy=json.loads((OUT/'threshold_analysis.json').read_text());primary=json.loads((OUT/'drift_result.json').read_text());r=json.loads((OUT/'retraining_metrics.json').read_text());v=policy['normal_validation'];t=policy['normal_test'];rec=policy['recommended'];false=policy['false_alarms'];docs=ROOT/'docs/aiops'
 normal=json.loads((OUT/'normal_daily.json').read_text())['test'];drift=json.loads((OUT/'drift_daily.json').read_text());pairs=[(a,b) for a,b in zip(normal,drift) if a['date']>=primary['start_date']]
 actual=np.array([a['actual'] for a,b in pairs]);changed=np.array([b['actual'] for a,b in pairs]);predicted=np.array([a['predicted'] for a,b in pairs])
 diagnostic={'normal':evaluate(actual,predicted),'injected':evaluate(changed,predicted),'normal_absolute_error_sum':float(np.abs(actual-predicted).sum()),'injected_absolute_error_sum':float(np.abs(changed-predicted).sum()),'normal_actual_sum':float(actual.sum()),'injected_actual_sum':float(changed.sum()),'mean_signed_normal_error_predicted_minus_actual':float((predicted-actual).mean())}
 (OUT/'injection_error_diagnostics.json').write_text(json.dumps(diagnostic,indent=2))
 sensitivity=[]
 for factor in [.20,.25]:
  result,_,_=simulate(factor);sensitivity.append({k:value for k,value in result.items() if 'records' not in k})
 (OUT/'sensitivity_analysis.json').write_text(json.dumps(sensitivity,indent=2))
 # Distribution values are recomputed from real rows, not inferred from summary statistics.
 with (OUT/'normal_rolling_wape.csv').open() as f:rows=list(csv.DictReader(f))
 vals={split:[float(row['rolling_wape']) for row in rows if row['split']==split and row['rolling_wape']] for split in ['validation','test']}
 fig,ax=plt.subplots(figsize=(10,4));ax.hist(vals['validation'],bins=12,alpha=.55,label='Normal Validation');ax.hist(vals['test'],bins=12,alpha=.55,label='Normal Test')
 for value,label,color,style in [(v['mean'],'Validation mean','#64748b',':'),(v['p90'],'Warning P90','#d97706','--'),(v['p95'],'Drift P95 / Gate','#dc2626','--'),(10,'Old drift 10%','#111827',':')]:ax.axvline(value,label=f'{label}: {value:.2f}%',color=color,ls=style)
 ax.set(title='Real daily rolling 7-day WAPE / frozen Validation calibration',xlabel='WAPE (%)',ylabel='Evaluation days');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'threshold_distribution.png',dpi=180);plt.close(fig)
 records=primary['drift_records'];base=primary['normal_records'];dates=[datetime.fromisoformat(x['date']) for x in records];values=[np.nan if x['rolling_wape'] is None else x['rolling_wape'] for x in records];normalvals=[np.nan if x['rolling_wape'] is None else x['rolling_wape'] for x in base]
 fig,ax=plt.subplots(figsize=(12,4));ax.plot(dates,normalvals,label='Original frozen v2 forecasts / normal actual');ax.plot(dates,values,label='Same frozen forecasts / +15% significant actual');ax.axvline(datetime.fromisoformat(primary['start_date']),label='Injection starts',color='black',ls=':');ax.axhline(rec['warning'],label='Validation P90 warning',color='#d97706',ls='--');ax.axhline(rec['drift'],label='Validation P95 drift',color='#dc2626',ls='--');ax.set(title='Primary +15% structural shift: NOT DETECTED' if not primary['detection_date'] else 'Primary structural shift',ylabel='Rolling 7-day WAPE (%)',xlabel='Daily evaluation after complete actual');ax.legend(fontsize=8);fig.autofmt_xdate();fig.tight_layout();fig.savefig(OUT/'normal_vs_drift_wape.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(2,1,figsize=(12,5),sharex=True);axes[0].plot(dates,values,label='Primary rolling WAPE');axes[0].axhline(rec['drift'],ls='--',color='red',label='Frozen drift threshold');axes[0].axvline(datetime.fromisoformat(primary['start_date']),ls=':',color='black');axes[0].set(ylabel='WAPE (%)',title='Daily detection timeline / no automatic detection or retrain');axes[0].legend(fontsize=8);axes[1].step(dates,[x['consecutive_drift_days'] for x in records],where='mid',label='Consecutive daily exceedances');axes[1].axhline(3,color='red',ls='--',label='3-day requirement');axes[1].set(ylabel='Days',ylim=(-.2,3.5));axes[1].legend(fontsize=8);fig.autofmt_xdate();fig.tight_layout();fig.savefig(OUT/'drift_detection_timeline.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4));values2=[r['production']['wape'],r['candidate']['wape']];bars=ax.bar(['Production / same Gate','Candidate / same Gate'],values2,color=['#94a3b8','#2563eb']);ax.axhline(r['gate_threshold'],label='Validation P95 absolute ceiling',ls='--',color='#dc2626');ax.set(title=f'MANUAL diagnostic / independent14-day Gate / {r["gate"]}',ylabel='WAPE (%)');ax.legend(fontsize=8)
 for bar,value in zip(bars,values2):ax.text(bar.get_x()+bar.get_width()/2,value+.5,f'{value:.3f}%',ha='center')
 fig.tight_layout();fig.savefig(OUT/'production_before_after.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,3));ax.axis('off');ax.text(.05,.6,f'Initial Production v{r["production_before"]}',fontsize=18);ax.text(.45,.6,'→',fontsize=24);ax.text(.55,.6,f'Current Production v{r["production_after"]}',fontsize=18);ax.text(.05,.25,f'Manual warm-start diagnostic → actual Gate {r["gate"]} → alias/stage promotion\nPrimary +15% detector: NOT_DETECTED; no automatic redeployment claimed',fontsize=12);fig.tight_layout();fig.savefig(OUT/'production_version_change.png',dpi=170);plt.close(fig)
 # Full loop diagram shows both the implemented automatic branch and actual manual execution.
 fig,ax=plt.subplots(figsize=(13,3));ax.axis('off');labels=['Normal Test','+15% injection','No 3-day detection','Manual diagnostic','Independent Gate','Production v2']
 for i,label in enumerate(labels):
  ax.text(i,.6,label,ha='center',va='center',bbox={'boxstyle':'round,pad=.5','facecolor':'#fff7ed' if i in [2,3] else '#eff6ff','edgecolor':'#94a3b8'},fontsize=10)
  if i<5:ax.annotate('',xy=(i+.72,.6),xytext=(i+.25,.6),arrowprops={'arrowstyle':'->'})
 ax.set(xlim=(-.7,5.7),ylim=(0,1));fig.tight_layout();fig.savefig(OUT/'experiment_timeline.png',dpi=180);plt.close(fig)
 metadata=client().get_model_version_by_alias('RenewableEnergyPredictor','Production');run=client().get_run(metadata.run_id)
 (OUT/'mlflow_verification.json').write_text(json.dumps({'production_version':str(metadata.version),'stage':metadata.current_stage,'run_id':metadata.run_id,'params':run.data.params,'metrics':run.data.metrics,'tags':run.data.tags},indent=2))
 threshold=f'''# 실제 정상 분포 기반 운영 임계값

Production 대상은 정확도 best naive가 아니라 AIOps 교육 목표에 따라 LSTM v2이다. v2 원본 Test WAPE42.860109%, best naive33.892438%의 한계를 숨기지 않는다.

매일 midnight-issued24-hour forecast의 실제24시간이 모두 확보된 뒤 한번만 최근7일168개 **비중복** target을 평가한다. Validation/Test 각각47개 완전한 daily forecast,41개 rolling 평가다. Validation 정상 분포로 Warning=P90={rec['warning']:.6f}%, Drift=P95={rec['drift']:.6f}%를 확정한 뒤 Test/injection을 평가했다. std는 population standard deviation이다. 원래 Validation은 v2 EarlyStopping에도 쓰인 구간이므로 완전히 새 calibration holdout은 아니며 추가 미래기간 검증이 필요하다.

| 분포 | mean | median | std | P75 | P90 | P95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''+''.join('| '+name+' | '+' | '.join(f'{stats[k]:.6f}' for k in ['mean','median','std','p75','p90','p95','min','max'])+' |\n' for name,stats in [('Validation',v),('Test',t)])+f'''
정상Test41회 중 기존10%초과={policy['old_policy_test_days_at_or_above_10']}, 기존8%초과={policy['old_gate_test_days_above_8']}. 새 Warning 발생{false['warning_days']}일, Drift 초과{false['drift_exceed_days']}일,3일연속 episode 오탐{false['drift_events']}회이다. Rolling windows가 겹치므로41개를 독립 표본으로 해석하지 않는다.

WAPE≥Drift를 달력상 연속된3개 **날짜**에서 확인해야 drift가 된다. 같은날actual 재전송은 멱등이며 상태를 증가시키지 않는다. 누락된날은 연속조건/rolling window를 끊고, 한 episode당3일째 한번 event를 발생시킨다.

절대Gate=P95={rec['gate']:.6f}% AND Candidate WAPE<Production WAPE를 동일한 independent Gate에 적용한다. P95는 오차가 큰 v2의 **데모용 운영 ceiling**이며 business accuracy SLA가 아니다. Calibration7일daily와 Gate14일rolling metric은 동일 WAPE 단위지만 가중치와길이가 다르므로 이 ceiling의 일반적 통계 보장은 주장하지 않는다. 모델별/계절별 calibration과 운영정책 추가검토가 필요하다. threshold를 injection을 보고 낮추지 않았다.
'''
 (docs/'threshold_policy.md').write_text(threshold)
 (docs/'drift_simulation_design.md').write_text(f'''# 고정 +15% 구조적 Drift 실험

시나리오: 신규 태양광 설비 편입으로 거래량 규모가 증가. 첫 완전Test forecast일~마지막날의 중간 날짜 **{primary['start_date']}**부터 actual만1.15배로 변경한다. Train peak의1%={primary['minimum_mwh']:.8f} MWh보다 큰 관측에만 적용한다. 이는 의미있는발전량의 사전 고정 휴리스틱이며 정확한 일출/일몰 classifier가 아니다. 작은 야간값과0은 그대로 유지한다. 원본CSV/모델/기존예측은 변경하지 않는다.

동일 이후 기간 원본 WAPE={primary['normal_wape_same_after_period']:.6f}%, 주입후={primary['drift_wape_after']:.6f}%. 주입전 기간 WAPE={primary['normal_wape_before']:.6f}%는 다른 기간이라 그것과의 단순 비교만으로drift효과를 주장하지 않는다.

첫threshold초과={primary['first_exceed_date']},2일연속={primary['two_consecutive_date']},3일확정={primary['detection_date']},delay={primary['detection_delay_days']}. None은 미발생을 뜻한다. **+15%는 탐지되지 않았고 자동재학습/자동배포는 실행하지 않았다.**

주입은 실제업무상의가정을 적용한 명시적simulation이지 새 관측데이터라고 주장하지 않는다. 정상reference 예측을 고정한실험과 현재Production캐시를 사용하는live API를 구분한다. +20/+25는 sensitivity_analysis.json에 별도로 기록했고 threshold나 primary15%를 바꾸지 않았다.

same-after normal L1 numerator={diagnostic['normal_absolute_error_sum']:.6f}, 주입후={diagnostic['injected_absolute_error_sum']:.6f}; actual denominator={diagnostic['normal_actual_sum']:.6f}→{diagnostic['injected_actual_sum']:.6f}. WAPE는 상대비율이므로 actual규모증가가 반드시WAPE증가로 이어지지 않는다. 이를 탐지성공으로 바꾸어기록하지 않는다.
''')
 (docs/'retraining_policy.md').write_text(f'''# Warm-start 및 독립 Deployment Gate

실제기본시나리오가미탐지되어 자동실행은없다. 별도 `{r['reason']}` 수동진단으로 실제weights/metrics/Gate/승격만검증했다. as-of={r['as_of']}, 미래데이터사용없음.

최근56일: {r['split']['known_start']}~{r['split']['gate_end']}.
앞42일fine-tuning: optimizer fit 첫35일(~{r['split']['fit_end']}), 내부EarlyStopping 마지막7일({r['split']['early_stopping_start']}~{r['split']['fine_tuning_end']}). Gate targets 마지막14일 {r['split']['gate_start']}~{r['split']['gate_end']}는fit/ES에 사용하지 않았다. Gate의 첫input168시간은 이미알려진이전context이므로앞42일과input만겹칠수있고label은겹치지않는다.

Production v{r['production_before']} weights를clone하고학습전 np.array_equal로동일성을확인했다. 기존Train-fitted scaler_v2를유지했다. LR1e-4, max20epoch, ES patience4/best weights restore, 실제{r['epochs_completed']}epochs, shuffle=False, seed42.

Gate는동일변경환경에서알려진과거168시간을두모델에똑같이제공한다. primary검출실험의고정원본prediction과Gate의현재관측input replay는별도목적이다. MWh복원/0floor/metrics는공용함수재사용.

Production MAE={r['production']['mae']:.6f}, RMSE={r['production']['rmse']:.6f}, WAPE={r['production']['wape']:.6f}%. Candidate MAE={r['candidate']['mae']:.6f}, RMSE={r['candidate']['rmse']:.6f}, WAPE={r['candidate']['wape']:.6f}%. 절대상한{r['gate_threshold']:.6f}%와상대개선을모두확인한실제결과={r['gate']}.

PASS만새Registryversion및Productionalias를승격하고캐시를clear한다. FAIL은Run/모델artifact를보존하며기존version을유지한다. 이번실제Run은PASS였고FAIL처리는controlled unit check 및source branch검증이다. 운영중FAIL이실제로발생했다고쓰지않는다. Gate평가기간을다음튜닝의validation으로재사용하면독립성보장이약해지므로반복실험/후속운영은새chronological holdout을확보해야한다.
''')
 troubleshooting=f'''# AIOps 운영 정책 트러블슈팅

## 사례 1. 임의8%/10%와실제성능분포불일치
문제/증상: v1=52.288744%, v2=42.860109%,best naive=33.892438%. 정상rolling Test mean={t['mean']:.6f}%, 기존10%는41회전부초과.
원인: 정상분포확인전정한운영가정.
영향: 정상모델도지속경보.
해결: injection전에Validation P90/P95로freeze.
전후: Warning={rec['warning']:.6f}%,Drift={rec['drift']:.6f}%, 정상Test3일episode오탐{false['drift_events']}회.
배운점: 임계값은정상운영분포에서출발해야한다. 높은threshold는미탐지를늘릴수있으므로정확도SLA와구분한다.

## 사례 2. LSTM이naive보다나쁜성능
문제/증상: v2 WAPE42.86%>naive33.89%.
원인: [기존진단](../troubleshooting_model_quality.md)의가설은아직인과검증되지않음.
영향: 높은정상오차로작은구조변화감지어려움.
해결: 이번은사용자요청대로LSTM v2운영loop,naive는benchmark유지.
전후: 정확도최고모델선택이라고포장하지않음.
배운점: 교육용운영대상선택과정확도기준선택은별개.

## 사례 3. +15%가WAPE상승/탐지로이어지지않음
문제/증상: same-after WAPE {primary['normal_wape_same_after_period']:.6f}%→{primary['drift_wape_after']:.6f}%,확정일=None.
원인: 상대오차분모도커지고각시간의과소/과대예측방향에따라residual변화가다름. 실제numerator/denominator는injection_error_diagnostics.json에기록.
영향: 고정P95정책은15%설비편입을미탐지.
해결: primary를FAIL_NOT_DETECTED로보존. threshold/15%를바꾸지않고20/25%별도민감도분석.
전후: 자동재학습없음;수동진단과분리.
배운점: 설비규모변화를WAPE하나로항상감지할수없다. Capacity-normalized/bias/level변화지표는후속후보이며이번에새검출기로구현하지않음.

## 사례 4. 8주전체fit+Gate중복누수방지
문제: 같은데이터로candidate를학습/검증하면낙관평가.
원인: 단일최근window를역할구분없이재사용.
영향: 안전배포판정왜곡.
해결: first42tuning(35fit+7ES),last14Gate;as-of까지만cut.
전후: Gate label은fit/ES와분리,actual weights equality 및split검사PASS.
배운점: Gate를EarlyStopping에도쓰면독립검증이아님.

## 사례 5. Gate FAIL의안전한기존Production유지
문제: 자동재학습성공을무조건배포로혼동할수있음.
원인: fitting과운영승격의역할혼동.
해결: absoluteP95 ceiling AND same-window relative improvement.
실제결과: 수동진단Production {r['production']['wape']:.6f}%→Candidate {r['candidate']['wape']:.6f}%,{r['gate']}, v{r['production_before']}→v{r['production_after']}.
FAIL branch는pureGate/alias불변unit check만실행했고실제FAIL training이일어났다고주장하지않는다.
배운점: Run/artifact보존과Production승격은별도.

## 사례 6. MLflow version형식검사오류
실제증상: registered_model.aliases['Production']와str version비교AssertionError(checks_failed.log).
원인: 설치된MLflow가aliasversion을int로반환.
해결: 비교전에str로정규화. 재실행API/Registry체크PASS.
영향: 첫검사가실패했으나Run/모델결과는조작하지않음.

## 설계상추가기록
- Git저장소가없어original_hashes 및new-file diff로기존모델/결과/문서보존검증.
- single-process RLock+atomic JSON, one-worker만지원. 멀티프로세스운영은transactional storage가후속요구사항.
- forecast/actual날짜불일치, duplicateforecast, conflictingactuals, out-of-order actuals를거절. actual재전송은daily count를늘리지않음.
- live자동재학습은56일의이미관측된history가없으면defer하며미래CSV로몰래채우지않는다.
- API contract check는현재v2(version2)로Gate기간날짜를replay한기능검사다. **후행weights로과거운영성능을검증했다고주장하지않음**. Threshold/Test및Gate의시간누수검증은별도의frozen실험과독립Gate기록으로평가했다.
- 기존Baseline v1 /predict앱은보존;AIOps앱은start_datetime필수및하루midnight forecast조건으로새entrypoint에분리.
- high threshold의무오탐은15%검출민감도성공을뜻하지않는다. Docker/dashboard용기능은준비됐으나primary자동loop검출데모는완료하지못했으므로최종ready=NO.
'''
 with (docs/'troubleshooting_aiops.md').open('a') as f:f.write('\n'+troubleshooting)
 (docs/'PRESENTATION_AIOPS_TROUBLESHOOTING.md').write_text(f'''# Troubleshooting 1 — 임의임계값은실제운영에서작동하지않았다

| Before | Observed | Action | After |
|---|---|---|---|
| Gate8%,Drift10% | 정상Test rollingmean {t['mean']:.2f}%;41/41이10%초과 | Validation정상P90/P95로calibration | Warning {rec['warning']:.2f}%,Drift {rec['drift']:.2f}%;정상3일오탐0회 |

Lesson: 운영정책은모델성능분포에서설계해야한다. 무오탐만으로민감도가좋다고할수없다.

# Troubleshooting 2 — 구조변화가항상오차율증가를뜻하지않았다

- 고정15%설비편입: 원본same-after WAPE {primary['normal_wape_same_after_period']:.2f}%→{primary['drift_wape_after']:.2f}%.
- 3일연속검출: **미탐지**. 자동재학습/배포없음. Threshold/주입률조작없음.
- 분모와residual방향의영향: WAPE만으로level shift에단조반응하지않음.
- 별도수동warm-start검증: Production {r['production']['wape']:.2f}% vs Candidate {r['candidate']['wape']:.2f}%, 독립Gate {r['gate']}, Registry v{r['production_before']}→v{r['production_after']}.
- 이번에검증한것: 코드/MLflow/안전Gate/일일상태. 아직검증못한것: 기본15%의자동검출성공.

Lesson: 모델이정상응답하고재학습가능해도업무drift를검출한다고가정하면안된다.

[정상분포](../../artifacts/aiops/threshold_distribution.png) · [실제drift비교](../../artifacts/aiops/normal_vs_drift_wape.png) · [수동Gate](../../artifacts/aiops/production_before_after.png)
''')
 (docs/'AIOPS_PIPELINE_SUMMARY.md').write_text(f'''# AIOps 운영파이프라인및실제실행요약

KPXCSV →168h/7features →ProductionLSTM v2 →24hforecast저장 →/actuals의24h도착 →daily rolling7-day WAPE →P90Warning/P95Drift →3 consecutive calendar days →warm-start →Candidate →independentGate →MLflowRegistry/Productionalias →version-keyedServingcache.

**실제primary15%는NOT_DETECTED**. 자동training/배포아님. 수동diagnostic에서Gate {r['gate']} 및v{r['production_before']}→v{r['production_after']}만별도검증했다. naivemodel은benchmark로만유지.

## 실행
```bash
.venv/bin/python -m pip install -r requirements-aiops.txt
.venv/bin/python scripts/analyze_aiops.py
.venv/bin/python scripts/run_aiops.py
.venv/bin/uvicorn serving_app.aiops_main:app --host 127.0.0.1 --port 8080 --workers 1
```
run_aiops는미탐지시명시적manualdiagnostic으로Candidate/Gate를검증하는실험runner이다. liveAPI자동분기는register_actuals의3일event및56일observedhistory만사용한다. 이미실행한동일holdout을계속반복튜닝하면독립성을약화시키므로새운영기간으로검증해야한다.

## API
- /health: 모델cache로드여부.
- /predict: start_datetime=첫과거관측의KST정각,sequence168개; forecast=start+168h가midnight. 저장된forecast는immutable. 반환24hour/mwh와정확한Registryversion.
- /actuals: date,actuals24개; forecast없는날/충돌/역순거절;동일actual재전송멱등.
- /aiops/status: 실제rollingWAPE,threshold,streak,version. 최소7개연속완전days이전은insufficient_data.
- /model/info: MLflowProductionalias의actualversion,metrics,params,featureconfig.
- /logs: 실제aiops.log.
- /batch-test: scenario=normal|drift; threshold/injection실험의초기v2reference예측으로고정된실제결과를replay한다. currentProductionversion도별도표시하며새모델의성능데모라고혼동하지않는다.

기존baseline v1앱은serving_app.main에서그대로사용가능하다. AIOps는새entrypoint를사용해기존168-only 계약을깨지않았다. 모델file/scaler/이전artifacts/문서는hash보존검증.

## 상태및남은작업
primary15%검출=미탐지,수동Gate={r['gate']},MLflow/API검사PASS. Docker/dashboard ready=NO(주요기본시나리오민감도미검증). 다음은Capacity/level/bias 지표의업무적적합성검토,새기간calibration/holdout,live동시성/데이터수집정책을정의하는것이다. Docker나dashboard는이번에구현하지않았다.
''')
 # Append, never falsify the primary detection failures.
 validation=json.loads((OUT/'validation.json').read_text());validation['required_docs_and_charts']='PASS';validation['MLflow_final_production_version']=str(metadata.version);validation['ready_for_docker_dashboard']='NO; primary drift scenario not detected'
 originals=json.loads((OUT/'original_hashes.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in originals.items())
 for folder in ['src','scripts','tests','serving_app']:
  for p in (ROOT/folder).rglob('*.py'):compile(p.read_text(),str(p),'exec')
 validation['syntax']='PASS';(OUT/'validation.json').write_text(json.dumps(validation,indent=2))
 sources=[p for base in ['src','scripts','tests','serving_app','docs/aiops'] for p in Path(base).rglob('*') if p.is_file() and '__pycache__' not in p.parts and str(p) not in originals]
 sources.append(Path('requirements-aiops.txt'))
 (OUT/'source_diff.patch').write_text(''.join(''.join(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile=str(p))) for p in sources))
 (OUT/'FILES.md').touch();files=sorted(p for p in OUT.rglob('*') if p.is_file());(OUT/'FILES.md').write_text('# AIOps 산출물\n\n'+''.join(f'- [{p.relative_to(OUT)}]({p.relative_to(OUT).as_posix()})\n' for p in files)+'\n## 추가소스/문서\n\n'+''.join(f'- [{p}]({"../../"+p.as_posix()})\n' for p in sources))
 print('Documents/charts generated; original files unchanged; primary15percent failures retained.')

if __name__=='__main__':main()
