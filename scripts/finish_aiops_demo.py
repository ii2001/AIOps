"""Append-only documentation and charts for the actual automatic Scenario2 execution."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,hashlib,difflib
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.config import ROOT


def main():
 parent=ROOT/'artifacts/aiops';out=parent/'scenario_2';docs=ROOT/'docs/aiops'
 result=json.loads((out/'demo_result.json').read_text());selected=result['selected'];r=result['candidate'];bias=json.loads((parent/'scenario_1_bias_analysis.json').read_text());first=json.loads((parent/'drift_result.json').read_text());threshold=json.loads((parent/'threshold_analysis.json').read_text())['recommended']
 strengths=[json.loads((out/f'output_curtailment_{p}.json').read_text()) for p in [15,20,25,30]];chosen=next(s for s in strengths if s['change_percent']==selected['change_percent']);records=chosen['records'];base=first['normal_records'];dates=[datetime.fromisoformat(x['date']) for x in records]
 def plot_series(ax,rows,label):ax.plot(dates,[np.nan if x['rolling_wape'] is None else x['rolling_wape'] for x in rows],label=label)
 fig,ax=plt.subplots(figsize=(11,4));plot_series(ax,base,'Original');plot_series(ax,first['drift_records'],'Capacity +15%');ax.axhline(threshold['drift'],color='red',ls='--',label='Frozen Drift P95');ax.set(title='Scenario1 preserved: +15% FAILED / NOT DETECTED',ylabel='Rolling7-day WAPE (%)');ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(parent/'scenario_1_failed_drift.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,4));plot_series(ax,base,'Original frozen LSTM-v2');plot_series(ax,records,f'Curtailment {selected["change_percent"]}%');ax.axvline(datetime.fromisoformat(selected['start_date']),ls=':',color='black',label='Injection start');ax.axvline(datetime.fromisoformat(selected['detection_date']),ls=':',color='purple',label='3-day detection');ax.axhline(threshold['warning'],color='orange',ls='--',label='Warning P90');ax.axhline(threshold['drift'],color='red',ls='--',label='Drift P95');ax.set(title='Scenario2: minimum detected strength in the tested grid',ylabel='Rolling7-day WAPE (%)');ax.legend(fontsize=8);fig.autofmt_xdate();fig.tight_layout();fig.savefig(parent/'scenario_2_drift_detection.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4));x=[abs(s['change_percent']) for s in strengths];w=[s['mean_wape_after'] for s in strengths];ax.plot(x,w,marker='o',label='Post-start pooled WAPE')
 for xx,ww,s in zip(x,w,strengths):ax.annotate('Detected' if s['detected'] else 'Not detected',(xx,ww),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
 ax.set(title='All curtailment sensitivity experiments / unchanged thresholds',xlabel='Actual reduction (%)',ylabel='WAPE (%)',xticks=x);ax.set_ylim(min(w)-4,max(w)+8);ax.legend();fig.tight_layout();fig.savefig(parent/'drift_sensitivity.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4));numbers=[r['production']['wape'],r['candidate']['wape']];bars=ax.bar(['Production','Candidate'],numbers,color=['#94a3b8','#2563eb']);ax.axhline(r['gate_threshold'],color='red',ls='--',label='Absolute ceiling');ax.set(title=f'Automatic independent Gate: {r["gate"]} / Production retained',ylabel='WAPE (%)',ylim=(0,max(numbers)*1.17));ax.legend()
 for b,n in zip(bars,numbers):ax.text(b.get_x()+b.get_width()/2,n+.4,f'{n:.3f}%',ha='center')
 fig.tight_layout();fig.savefig(parent/'retraining_before_after.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,3));ax.axis('off');labels=['Normal','Curtailment -20%','3-day detection','AUTO warm-start','Gate FAIL','Retain v1']
 for i,label in enumerate(labels):
  ax.text(i,.6,label,ha='center',bbox={'boxstyle':'round,pad=.5','facecolor':'#fff7ed' if i>=4 else '#eff6ff','edgecolor':'#94a3b8'},fontsize=10)
  if i<5:ax.annotate('',xy=(i+.7,.64),xytext=(i+.25,.64),arrowprops={'arrowstyle':'->'})
 ax.text(1,.23,selected['start_date'],ha='center');ax.text(2,.23,selected['detection_date'],ha='center');ax.text(3,.23,'42 fit /14 Gate',ha='center');ax.set(xlim=(-.7,5.7),ylim=(0,1));fig.tight_layout();fig.savefig(parent/'aiops_full_timeline.png',dpi=180);plt.close(fig)
 marker='[SCENARIO_2_AUTOMATIC]'
 append=f'''\n\n# {marker} 사례 6. +15% Drift에서 상대오차가 감소했다

**문제/실제:** 동일 이후 기간 WAPE {first['normal_wape_same_after_period']:.6f}% → {first['drift_wape_after']:.6f}%. 첫 실험은 FAIL_NOT_DETECTED, 자동 재학습 NOT_RUN으로 그대로 보존했다.
**실제 bias:** 평균 signed error(predicted-actual)={bias['mean_signed_error']:.6f} MWh, median={bias['median_signed_error']:.6f} MWh, overprediction={bias['overprediction_ratio']*100:.6f}%, underprediction={bias['underprediction_ratio']*100:.6f}%, MAE={bias['mae']:.6f} MWh.
**원인 해석:** 평균 양의 bias가 확인됐지만 모든 오차가 가까워졌다는 의미는 아니다. 이전 실험의 MAE/RMSE는 증가했고, actual 분모 증가와 오차 방향이 함께 작용해 WAPE가 감소했다.
**해결:** 업무적으로 가능한 반대 변화인 출력제어/설비 성능 저하를 별도 시나리오로 설계했다. first failure 기록/CSV/기존모델을 수정하지 않았다.
**배운 점:** 평균 bias·절대오차·상대오차 분모를 함께 봐야 한다.

# 사례 7. 기본 -15%와 모든 민감도 실험

시작일 {selected['start_date']}, 유의미한 발전량 조건={selected['minimum_mwh']:.8f} MWh(기존 Train peak1%, 첫 실험과 동일), 기존 threshold Warning={threshold['warning']:.6f}%,Drift={threshold['drift']:.6f}%를 유지했다.

| 감소율 | 이후 pooled WAPE | 최대 rolling WAPE | 첫 초과 | 3일 탐지 |
|---|---:|---:|---|---|
'''+''.join(f"| {s['change_percent']}% | {s['mean_wape_after']:.6f}% | {s['max_rolling_wape']:.6f}% | {s['first_exceed_date']} | {s['detection_date']} |\n" for s in strengths)+f'''
**기본 -15%는 미탐지**였다. 시험한 grid에서 최소 검출은 {selected['change_percent']}%; 연속 변화율 전체의 수학적 최소값이라고 주장하지 않는다. Test를 이용해 데모 강도를 선정한 민감도 실험이며 새로운 운영기간에 일반화된 검출성능이라고 주장하지 않는다.

# 사례 8. Detector가 실제로 재학습을 호출했다

첫 초과={selected['first_exceed_date']},2일={selected['two_consecutive_date']},3일={selected['detection_date']}, delay={selected['detection_delay']}일. /batch-test → replay_demo → check_and_trigger → daily_policy event → retrain 경로이며 CLI/사람이 retrain을 직접 호출하지 않았다. actual24시간을 하루 한 번 평가하며 동일 날짜로3회를 만들지 않았다.

**시간/Registry 분리:** 기존 수동 Production v2는 12월17일까지 fine-tuning/ES를 경험했으므로 12월8일 시나리오에 그대로 넣으면 시점 해석이 혼동될 수 있다. 원본 당시 Production LSTM-v2의 실제 MLflow version1 weights를 읽어 별도 `scenario_2/mlflow.db`에 초기 Production으로 복제했다. weights가 baseline_v2.keras와 같은지도 확인했다. 기존 root Registry 수동 v2와 기존 Run/artifact는 변경하지 않았다.

**데이터 반영:** 최근56일 {r['split']['known_start']}~{r['split']['gate_end']}; first42 fit={r['split']['fine_tuning_end']},last14 Gate={r['split']['gate_start']}~{r['split']['gate_end']}. 10epochs,LR1e-4,seed42,shuffle=False. 이번은 EarlyStopping 없이42일 전체를fit했고 Gate는 어떤 fitting/모델선택에도 사용하지 않았다. 이는 검출 초기에 최근 감소값이 내부ES에만 들어가는 상황을 피하기 위한 명시적 변경이다.

Optimizer fit에 실제 감소 관측 {r['retrain_data_contains_drift']['hours']}시간이 포함됐다: {r['retrain_data_contains_drift']['first']}~{r['retrain_data_contains_drift']['last']}. as-of 이후 미래값은 사용하지 않았다. Gate 입력 context는 과거fit구간과 겹칠 수 있지만 label은 겹치지 않는다. detector실험은 원본 prediction을 고정한actual-only 시뮬레이션이며 Gate에서는 두 모델에 동일한 관측된감소input을 제공했다.

# 사례 9. 자동 Candidate가 상대개선 조건을 통과하지 못했다

Production: MAE={r['production']['mae']:.6f},RMSE={r['production']['rmse']:.6f},WAPE={r['production']['wape']:.6f}%. Candidate: MAE={r['candidate']['mae']:.6f},RMSE={r['candidate']['rmse']:.6f},WAPE={r['candidate']['wape']:.6f}%.

실제 Gate={r['gate']}. Candidate는 절대상한 이하였지만 동일Gate의 Production보다 WAPE가 높아 거절됐다. 임계값/epochs를 변경해 PASS를 유도하지 않았다. 실제 [FAIL] candidate rejected → existing production retained 로그, Candidate Run/artifact 보존, isolated Production v{result['version_before']}→v{result['version_after']} 유지.

**API/cache:** 승격이 없으므로 기존 모델을 유지하는 것이 정상이다. API실제version,Registryversion,명시적모델예측이 일치했다. cache mismatch는 발생하지 않았고 이슈를 만들어 기록하지 않는다. Promotion/cache invalidation branch는 이번FAIL 실행에서는 실행하지 않았음을 명확히 한다.

**결론:** 자동 탐지→warm-start→독립Gate→안전한유지까지 end-to-end 제어루프는 검증됐다. 모델의 성능 복구/자동승격 성공은 아니다. Docker/dashboard/report의 다음 구현 단계는 진행 가능하지만 Candidate 품질 개선과 새기간검증은 남아 있다.
'''
 target=docs/'troubleshooting_aiops.md'
 if marker not in target.read_text():
  with target.open('a') as f:f.write(append)
 slides=f'''\n\n## {marker} 시행착오 2 — Drift를 넣었는데 오차율이 줄었다

- **Before:** 신규설비 +15%, WAPE51.13%→49.82%,미탐지.
- **Bias:** mean signed error +{bias['mean_signed_error']:.2f} MWh, 과대예측{bias['overprediction_ratio']*100:.2f}%.
- **Action:** 업무적으로 가능한 출력제어/설비성능저하. 기본-15%도미탐지, -20/-25/-30%를전부별도평가.
- **Result:** 시험한최소검출-20%, {selected['detection_date']} 3일확정(delay{selected['detection_delay']}일).
- **AUTO:** Detector가warm-start를호출. 최근6주fit/2주Gate, fit안에감소실제값{r['retrain_data_contains_drift']['hours']}시간.
- **Gate:** Production{r['production']['wape']:.2f}% vs Candidate{r['candidate']['wape']:.2f}%, **FAIL**. 모델승격없이기존Production유지.
- **Lesson:** 데이터를바꿨다는사실이아니라업무환경변화와모델오차방향을함께검토해야한다. 자동재학습성공도자동배포의근거가아니다.

별도scenario2 Registry를사용했고 이전수동v1→v2 승격은그대로보존했다. 이번새자동실험은isolated v1유지이며복구배포성공으로포장하지않는다.
'''
 target=docs/'PRESENTATION_AIOPS_TROUBLESHOOTING.md'
 if marker not in target.read_text():
  with target.open('a') as f:f.write(slides)
 (docs/'PRESENTATION_AIOPS_FINAL_FLOW.md').write_text(f'''# AIOps End-to-End — 실제 자동 탐지와 안전한 배포 거절

Normal → Daily rolling7-day WAPE → Persistent curtailment → 3-day exceed → Auto Drift Detection → Warm-start → Candidate → Independent Gate → FAIL → Production retained

| 실제 항목 | 결과 |
|---|---|
| First +15% scenario | FAIL_NOT_DETECTED, 보존 |
| 기본 -15% | 미탐지 |
| 최소 검출 강도(시험 grid) | -20% |
| 같은 이후 원본 WAPE | {selected['original_wape_after']:.6f}% |
| 감소 시나리오 WAPE | {selected['mean_wape_after']:.6f}% |
| threshold | {threshold['drift']:.6f}% (변경 없음) |
| first/second/third exceed | {selected['first_exceed_date']} / {selected['two_consecutive_date']} / {selected['detection_date']} |
| delay | {selected['detection_delay']}일 |
| auto retrain | YES, 실제 MLflow Run {r['run_id']} |
| Production Gate WAPE | {r['production']['wape']:.6f}% |
| Candidate Gate WAPE | {r['candidate']['wape']:.6f}% |
| Gate | FAIL, 상대 개선 실패 |
| isolated Registry version | v1→v1 RETAINED |
| original Registry | 이전 수동 Production v2 보존 |

모델성능 복구/자동승격은 성공하지 않았다. Detector→재학습→평가→안전한 거절 루프는 실제 자동 실행으로 확인했다. 후보성능 개선,새기간 holdout,검출민감도 일반화는 다음 과제다.

## 재현
`.venv/bin/python scripts/run_aiops_demo.py`
최초실행은분리DB/새Run/API자동루프를수행한다. 이미demo_result가있으면기록된결과를출력해기존실험을덮어쓰거나재학습하지않는다. 실행log: artifacts/aiops/scenario_2/execution.log. baseline-v1앱과원본파일은보존한다.

검증: `.venv/bin/python tests/check_scenario2.py` (새training없이실제Run/API/로그/시점기록을검증).

[민감도 차트](../../artifacts/aiops/drift_sensitivity.png) · [감지 차트](../../artifacts/aiops/scenario_2_drift_detection.png) · [Gate 비교](../../artifacts/aiops/retraining_before_after.png) · [전체 타임라인](../../artifacts/aiops/aiops_full_timeline.png)
''')
 oldlog=(out/'previous_aiops.log').read_text();canonical=parent/'aiops.log'
 if marker not in canonical.read_text():
  with canonical.open('a') as f:f.write('\n[INFO] '+marker+' isolated automatic experiment follows; original manual Registry unchanged\n'+(out/'aiops.log').read_text())
 assert canonical.read_text().startswith(oldlog)
 for name in ['troubleshooting_aiops.md','PRESENTATION_AIOPS_TROUBLESHOOTING.md']:assert (docs/name).read_bytes().startswith((out/('previous_'+name)).read_bytes())
 validation=json.loads((out/'validation.json').read_text());validation.update({'troubleshooting_appended':'PASS','presentation_docs_updated':'PASS','all_charts_generated':'PASS','ready_for_docker_dashboard_report':'YES; safe rejection loop, not model quality recovery'})
 (out/'validation.json').write_text(json.dumps(validation,indent=2))
 previous=json.loads((out/'previous_validation.json').read_text());current=json.loads((parent/'validation.json').read_text())
 for key,value in previous.items():assert current[key]==value
 current['scenario_2_validation']=validation;(parent/'validation.json').write_text(json.dumps(current,indent=2))
 originals=json.loads((out/'original_hashes.json').read_text());sources=[p for base in ['src','scripts','tests','serving_app'] for p in (ROOT/base).rglob('*.py') if '__pycache__' not in p.parts]
 changed=[p for p in sources if str(p.relative_to(ROOT)) not in originals or hashlib.sha256(p.read_bytes()).hexdigest()!=originals[str(p.relative_to(ROOT))]]
 for p in changed:compile(p.read_text(),str(p),'exec')
 (out/'source_diff.patch').write_text(''.join(''.join(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null' if str(p.relative_to(ROOT)) not in originals else 'previous/'+str(p.relative_to(ROOT)),tofile=str(p.relative_to(ROOT)))) for p in changed))
 (out/'FILES.md').touch();files=sorted(p for p in out.rglob('*') if p.is_file() and 'matplotlib-cache' not in p.parts);(out/'FILES.md').write_text('# Scenario2 自动실험 산출물\n\n'+''.join(f'- [{p.relative_to(out)}]({p.relative_to(out)})\n' for p in files)+'\n차트 및 추가 JSON은 상위 artifacts/aiops/에 보존. 기존 첫 실패 결과는 수정하지 않음.\n')
 print('Append-only documents and actual charts complete. Gate FAIL / retained preserved.')

if __name__=='__main__':main()
