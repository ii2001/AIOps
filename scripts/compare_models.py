"""Build comparisons/charts/documents solely from actual v1, naive and v2 evaluations."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import ROOT,INPUT_LEN
from src.benchmarks import comparison_data,seasonal_predictions
from src.metrics import evaluate


def main():
    out=ROOT/'artifacts/model_comparison'
    v1=json.loads((out/'v1_metrics.json').read_text());v2=json.loads((out/'v2_metrics.json').read_text());naive=json.loads((out/'naive_metrics.json').read_text())
    times,values,splits,history,actual=comparison_data()
    predictions={'Baseline v1':np.load(ROOT/'artifacts/test_predicted.npy'),**seasonal_predictions(history),'Baseline v2 LSTM':np.load(out/'v2_test_predicted.npy')}
    scores={name:evaluate(actual,p) for name,p in predictions.items()}
    assert all(np.isclose(scores['Baseline v1'][k],v1['metrics']['test'][k],rtol=1e-12) for k in ['mae','rmse','wape'])
    assert all(np.isclose(scores['Baseline v2 LSTM'][k],v2['metrics']['test'][k],rtol=1e-12) for k in ['mae','rmse','wape'])
    reference=scores['Baseline v1']['wape'];best=min(scores,key=lambda k:scores[k]['wape']);best_naive=min(naive,key=lambda k:naive[k]['wape'])
    with (out/'model_comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['model','mae_mwh','rmse_mwh','wape_percent','wape_improvement_vs_v1_percent'])
        for name,s in scores.items():writer.writerow([name,s['mae'],s['rmse'],s['wape'],(reference-s['wape'])/reference*100])
    print(json.dumps(scores,indent=2),flush=True)
    origin=splits[2][0]+INPUT_LEN;sample=next(i for i in range(len(actual)) if times[origin+i].hour==0)
    sample_rows=[]
    with (out/'forecast_comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['timestamp','actual','v1','best_naive','v2'])
        for h in range(24):
            row=[times[origin+sample+h].isoformat(),actual[sample,h],predictions['Baseline v1'][sample,h],predictions[best_naive][sample,h],predictions['Baseline v2 LSTM'][sample,h]]
            writer.writerow(row);sample_rows.append(row)
    fig,ax=plt.subplots(figsize=(10,4));labels=['LSTM v1','Previous Day','Previous Week','7-Day Mean','LSTM v2'];wapes=[s['wape'] for s in scores.values()];bars=ax.bar(labels,wapes,color=['#94a3b8','#60a5fa','#60a5fa','#60a5fa','#2563eb'])
    for bar,w in zip(bars,wapes):ax.text(bar.get_x()+bar.get_width()/2,w+.5,f'{w:.2f}%',ha='center',fontsize=10)
    ax.axhline(8,color='#16a34a',ls='--',label='Planned Gate 8%');ax.axhline(10,color='#d97706',ls=':',label='Planned drift reference 10%');ax.set(title='Identical test windows / 24-hour forecast WAPE',ylabel='WAPE (%)',ylim=(0,max(wapes)*1.18));ax.legend();fig.tight_layout();fig.savefig(out/'wape_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4));ax.plot(range(24),actual[sample],label='Actual',color='#0f172a',lw=2,marker='o')
    for name in ['Baseline v1',best_naive,'Baseline v2 LSTM']:ax.plot(range(24),predictions[name][sample],label=name,marker='.')
    ax.set(title=f'{times[origin+sample].date()} / same real test day / Jeollanam-do solar',xlabel='KST hour (interval start)',ylabel='MWh',xticks=range(0,24,2));ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'forecast_comparison.png',dpi=180);plt.close(fig)
    # Nonoverlapping midnight 24h forecasts make the operational seven-day discussion explicit.
    daily=[i for i in range(len(actual)) if times[origin+i].hour==0]
    weekly={}
    for name,p in predictions.items():
        windows=[evaluate(actual[daily[j:j+7]],p[daily[j:j+7]])['wape'] for j in range(len(daily)-6)]
        weekly[name]={'convention':'consecutive midnight-issued 24h forecasts, 7 days / 168 disjoint target hours', 'windows':len(windows),'wapes':windows,'min':min(windows),'max':max(windows),'at_or_above_10':sum(w>=10 for w in windows)}
    (out/'rolling_7day_diagnostics.json').write_text(json.dumps(weekly,indent=2))
    fig,ax=plt.subplots(figsize=(14,4));ts=[times[origin+i+h] for i in daily for h in range(24)];a=actual[daily].reshape(-1);p=predictions['Baseline v2 LSTM'][daily].reshape(-1);ax.plot(ts,a,label='Actual',lw=1);ax.plot(ts,p,label='LSTM v2 / midnight-issued 24h',lw=1,alpha=.8);ax.set(title='Test-period daily 24-hour forecasts (nonoverlapping targets)',xlabel='KST timestamp',ylabel='MWh');ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(out/'actual_vs_v2_test.png',dpi=170);plt.close(fig)
    hourly={str(h):float(np.mean([v for t,v in zip(times,values) if t.hour==h])) for h in range(24)}
    monthly={str(m):float(np.mean([v for t,v in zip(times,values) if t.month==m])) for m in range(1,13)}
    lead_scores={str(h+1):evaluate(actual[:,h],predictions['Baseline v1'][:,h]) for h in range(24)}
    (out/'v1_diagnostics.json').write_text(json.dumps({'hourly_mean_mwh':hourly,'monthly_mean_mwh':monthly,'v1_by_lead_hour':lead_scores},indent=2))
    second=scores['Baseline v2 LSTM']['wape'];bn=naive[best_naive]['wape'];improvement=(reference-second)/reference*100
    decision='시간 feature 추가와 capacity 조정으로 baseline 품질 개선 성공' if second<bn and second<reference else '모델 개선 효과는 있었으나 단순 seasonal baseline을 아직 이기지 못함' if second<reference else '시간 feature 및 capacity 조정만으로는 개선되지 않음'
    conclusion={'best_model':best,'best_wape':scores[best]['wape'],'best_naive':best_naive,'v2_vs_v1_improvement_percent':improvement,'v2_vs_best_naive':'BETTER' if second<bn else 'WORSE' if second>bn else 'TIE','decision':decision,'gate_8_gap_percentage_points':scores[best]['wape']-8,'drift_10_gap_percentage_points':scores[best]['wape']-10,'aiops_ready':'NOT YET' if scores[best]['wape']>8 else 'QUALITY_GATE_ONLY_PASSED; operational drift validation still required'}
    (out/'decision.json').write_text(json.dumps(conclusion,ensure_ascii=False,indent=2))
    table='| Model | MAE MWh | RMSE MWh | WAPE % | v1 대비 개선율 % |\n|---|---:|---:|---:|---:|\n'+''.join(f"| {name} | {s['mae']:.6f} | {s['rmse']:.6f} | {s['wape']:.6f} | {(reference-s['wape'])/reference*100:.6f} |\n" for name,s in scores.items())
    (out/'benchmark_summary.md').write_text('# 실제 동일-window 모델 비교\n\n'+table+f'\n최저 WAPE: **{best}**.\n')
    scope=f'''평가 범위: v1과 동일한1123 rolling origins ×24 horizons. 첫 target {times[origin].isoformat()}, 마지막 horizon {times[-1].isoformat()}. ground truth는 v1 test_actual.npy를 그대로 사용해 float32 normalization/inverse 오차까지 일치시켰다. 이전일은 입력 마지막24시간, 이전주는 입력 첫24시간, 평균은 입력7×24시간의 같은 위치 평균으로 계산했으며 모든 참조는 forecast origin 이전이다. 원시70/15/15 split과 파티션 내부 sequence, Train-only scaler 및 출력0 floor 정책을 유지했다.'''
    doc=f'''# 모델 성능 트러블슈팅 - Baseline v1

## 1. 문제 발견
v1은168시간 입력→24출력, FastAPI200 및validation이 동작했지만 Test MAE={scores['Baseline v1']['mae']:.6f} MWh, RMSE={scores['Baseline v1']['rmse']:.6f} MWh, WAPE={reference:.6f}%였다. 기능과 예측 품질은 별개다.

## 2. 왜 문제인가
기획 Gate WAPE≤8%, 최근7일 drift WAPE≥10% 대비 초기 baseline 오차가 높다. 코드는 정상인데 제안된 품질 기준에서는 운영 불가능한 상태다. 전체 Test WAPE만으로 개별 최근7일 경보를 단정하지 않고 실제 daily-origin7일 집계도 별도로 계산했다. 운영 threshold를 코드에 적용하거나 변경하지 않았다.

## 3. 기준 모델 비교
{table}
{scope}

{'복잡한 LSTM v1이 세 단순 seasonal baseline보다 성능이 낮았다.' if all(s['wape']<reference for s in naive.values()) else '각 기준모델의 실제 결과는 위 표와 같다.'}

## 4. 원인 가설
확인된 코드 사실: v1 입력은 MWh 하나, 명시적 hour/weekday/season/weather/capacity feature가 없다. 24개 직접출력 Dense 구조와 LSTM32를30epochs 학습했다. 실제 전체 시계열의 시간대 평균 min={min(hourly.values()):.6f}, max={max(hourly.values()):.6f} MWh로 시간대 패턴이 존재한다. 월별 평균 min={min(monthly.values()):.6f}, max={max(monthly.values()):.6f} MWh도 변하지만 이것만으로 weather/설비/출력제어의 인과관계를 확인할 수는 없다. [실제 진단값](../artifacts/model_comparison/v1_diagnostics.json).

가능한 원인: 시간대·요일·연중 계절 정보를 명시하지 않아 주기를 학습하기 어려웠을 수 있다. weather/설비규모/출력제어 등 외생변수 부재가 일별 변동 오차를 키웠을 수 있다. direct24출력의 학습량이나 capacity 부족도 가능하지만 확정 원인은 아니다. Train이1월~9월, Test가11~12월이므로 계절 일반화 문제도 가설이다. 야간 관측값을 임의0으로 바꾸지 않았다.

## 5. 개선 조치
MWh + hour sin/cos + weekday sin/cos + day-of-year sin/cos의7features, LSTM64→Dense64(ReLU)→Dense24를 별도 v2로 학습했다. Seed42, shuffle=False, 최대60epochs, val_loss EarlyStopping patience10/best weights restore. 실제 종료 {v2['epochs_completed']}epoch, best epoch {v2['best_epoch']}. sin/cos는 재fit하지 않으며 MWh scaler는 Train-only이다. 미래 날씨/target 시각 feature는 사용하지 않았다.

시간 feature, capacity, 학습 종료조건이 함께 바뀌었으므로 결과를 **시간 feature만의 순수 효과**로 인과 해석할 수 없다. 한 번의 고정-seed 실험이며 하이퍼파라미터 탐색은 수행하지 않았다.

## 6. 개선 전후 결과
v1 WAPE={reference:.6f}%, v2={second:.6f}%, 상대 개선율={improvement:.6f}%. Best naive={best_naive} / {bn:.6f}%.

## 7. 결론
{decision}. 전체 최저 WAPE 모델은 {best} / {scores[best]['wape']:.6f}%이다. 제안 Gate8%까지 {scores[best]['wape']-8:.6f} percentage points, drift10%까지 {scores[best]['wape']-10:.6f} percentage points 차이가 있다.
Best model의 daily-origin7일 평가 {weekly[best]['windows']} windows 중 {weekly[best]['at_or_above_10']}개가10% 이상이고, 범위는{weekly[best]['min']:.6f}~{weekly[best]['max']:.6f}%였다. 현재 정의의 경보가 초기 상태부터 빈번할 수 있으므로 정상 조건/최근7일 집계 규약과 운영 정책 검증이 필요하다. [실제7일 집계](../artifacts/model_comparison/rolling_7day_diagnostics.json). 운영 준비 판정: {conclusion['aiops_ready']}.

## 8. 다음 개선 방향
우선 weather(일사량/운량)·설비용량·출력제어 관련 데이터의 접근성과 timestamp 품질을 확인하고, 현재 strongest naive를 기준으로 개선을 검증한다. 동일capacity/동일학습설정으로 시간feature ablation을 하면 기여도를 더 정확히 구분할 수 있다. XGBoost/LightGBM 등 lag+calendar 기반 비교도 후속 후보이며 Transformer/TCN 확장은 단순모델과 외생변수 검증 이후에 판단한다. 이번에는 수집/구현하지 않았다. Test를 후보 비교에 사용했으므로 Production 선택 전 새 기간의 독립 holdout이나 walk-forward 검증이 필요하다.
'''
    (ROOT/'docs/troubleshooting_model_quality.md').write_text(doc)
    slide=f'''# Troubleshooting: 모델은 동작했지만 예측 품질이 낮았다

### 문제
- API168→24,200/422 정상. v1 WAPE **{reference:.2f}%**.
- {'세 seasonal naive보다 낮은 성능.' if all(s['wape']<reference for s in naive.values()) else 'naive와 공정 비교 필요.'}

### 원인 가설
MWh 단일feature, 명시적 시간/요일/계절 정보 부재, weather/설비용량 부재. 인과 원인은 아직 검증하지 않음.

### 개선
Cyclical time features 6개 + LSTM64/Dense64 + EarlyStopping. v1 원본과 동일Test1123×24 유지.

### 결과
- v1 WAPE: **{reference:.2f}%**
- Best Naive ({best_naive}): **{bn:.2f}%**
- v2 WAPE: **{second:.2f}%**
- v1 대비 상대 개선율: **{improvement:.2f}%**
- {decision}

| 단계 | 내용 |
|---|---|
| Before | LSTM v1 WAPE {reference:.2f}% |
| 발견 | Best naive WAPE {bn:.2f}%로 v1보다 {'우수' if bn<reference else '낮음'} |
| 원인 가설 | 명시적 시간정보와 외생변수 부재 |
| Action | Cyclical features + LSTM v2 + EarlyStopping |
| After | v2 WAPE {second:.2f}%, relative improvement {improvement:.2f}% |
| Next | Weather/Capacity 데이터 품질 검토 + naive 대비 재검증 |

### 배운 점
“모델이 정상 응답한다고 운영 가능한 것은 아니다. AIOps의 출발점은 모델 품질 기준을 먼저 검증하는 것이다.”

Gate8%/drift10%는 이번에 적용하지 않았다. best model WAPE {scores[best]['wape']:.2f}%로 현재 운영 준비 판정은 {conclusion['aiops_ready']}이다. feature와capacity/학습설정이 함께 바뀐 단일실험이므로 시간정보만의 효과로 단정하지 않는다.

[WAPE 비교](../artifacts/model_comparison/wape_comparison.png) · [실제 하루 forecast](../artifacts/model_comparison/forecast_comparison.png).
'''
    (ROOT/'docs/PRESENTATION_TROUBLESHOOTING.md').write_text(slide)
    (ROOT/'docs/model_selection.md').write_text('# 실제 모델 후보 선택\n\n'+table+f'\nBest model: {best}; best naive: {best_naive}. 주요지표 WAPE에 따라 선정했으며 LSTM이라는 이유로 우선하지 않았다.\n\nProduction candidate 추천: **현재는 운영용 승격 추천하지 않음**. Best WAPE {scores[best]["wape"]:.6f}%와 계획 Gate8%를 비교해야 하며, 독립 미래기간/walk-forward 검증도 필요하다. 현재 {best}를 차기 개선의 성능 기준으로 사용한다. 상대적으로 가장 좋다는 의미와 운영 기준 통과는 다르다.\n\n'+scope+'\n\n'+decision+'\n')
    print('Decision:',json.dumps(conclusion,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
