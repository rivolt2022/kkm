"""
국민대학교 AI빅데이터 분석 경진대회 - 공행성 쌍 판별 및 무역량 예측
Baseline 개선 버전: XGBoost 회귀 모델 및 향상된 feature engineering 사용
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
import warnings
import argparse
import sys
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# evaluation.py import를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../document'))
from evaluation import comovement_score, comovement_f1, comovement_nmae


def safe_corr(x, y):
    """안전한 상관계수 계산 (표준편차가 0인 경우 처리)"""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def robust_lagged_corr(x, y, lag, top_k=2, use_log=True):
    """
    lag 기준으로 정렬된 두 시계열 x, y에 대해
    - (필수) lag를 맞춰 align 한 뒤
    - (옵션) log1p 변환을 하고
    - 상위 top_k 스파이크(값이 큰 달들)를 제거한 후
    상관계수를 다시 계산하는 함수.

    스파이크가 몇 개 빠지면 corr가 확 죽는 쌍을 걸러내는 용도.

    Args:
        x, y : 1D numpy array (leader, follower 원 시계열)
        lag  : int, leader_t vs follower_{t+lag} 를 비교할 lag
        top_k: int, 스파이크로 간주해서 제거할 상위 구간 개수
        use_log: bool, 상관계수 계산을 log1p 스케일에서 할지 여부
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # lag 맞춰서 정렬 (네 원래 safe_corr 쓸 때와 동일한 방식)
    x_aligned = x[:-lag]
    y_aligned = y[lag:]
    n = len(x_aligned)
    if n == 0:
        return 0.0

    if use_log:
        x_aligned = np.log1p(x_aligned)
        y_aligned = np.log1p(y_aligned)

    # 데이터가 너무 적으면 robust하게 제거하는 것 자체가 불안하니 그냥 corr 리턴
    if n <= top_k + 3:
        return safe_corr(x_aligned, y_aligned)

    # "스파이크" 정의: |x| + |y| 값이 큰 순서대로 top_k 개
    spike_score = np.abs(x_aligned) + np.abs(y_aligned)
    spike_idx = np.argsort(-spike_score)[:top_k]

    mask = np.ones(n, dtype=bool)
    mask[spike_idx] = False

    # 스파이크 제거 후 corr
    return safe_corr(x_aligned[mask], y_aligned[mask])


def find_comovement_pairs(
    pivot,
    max_lag=6,
    min_nonzero=12,
    corr_threshold=0.35,
    use_log_corr=True,     # log1p 스케일에서 corr 계산할지
    top_k_spike=2,         # 스파이크로 제거할 상위 구간 개수
    use_robust=True,       # 스파이크 제거한 corr로 threshold 비교할지
):
    """
    공행성 쌍 탐색 (스파이크 기반 공행성 필터링 포함 버전)

    - 각 (A, B) 쌍에 대해 lag = 1 ~ max_lag까지 Pearson 상관계수 계산
      (기본은 log1p(value) 스케일에서)
    - 절댓값이 가장 큰 상관계수와 lag를 선택 (best_lag, base_corr)
    - 그 lag에 대해 스파이크 몇 개 제거한 robust corr를 추가로 계산
    - |robust_corr| >= corr_threshold 인 쌍만 공행성쌍으로 채택

    Args:
        pivot        : item_id × ym 피벗 테이블 (value 기준)
        max_lag      : 최대 lag (월 단위)
        min_nonzero  : 최소 비제로 개수 (너무 sparse 한 시계열은 배제)
        corr_threshold: robust corr 절댓값 기준 임계치
        use_log_corr : True면 log1p 스케일에서 corr 계산
        top_k_spike  : robust corr 계산 시 제거할 스파이크 개수
        use_robust   : True면 robust_corr 기준으로 threshold 비교,
                       False면 base_corr 기준으로 비교 (fallback 용)

    Returns:
        pairs: DataFrame
            columns = [leading_item_id, following_item_id, best_lag, max_corr]
            max_corr에는 "threshold를 만족한 corr" (기본: robust_corr)을 저장
    """
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)

    results = []

    for i, leader in tqdm(enumerate(items), total=len(items), desc="공행성 쌍 탐색"):
        x_raw = pivot.loc[leader].values.astype(float)
        if np.count_nonzero(x_raw) < min_nonzero:
            continue

        # 미리 log 변환 버전도 만들어두기
        if use_log_corr:
            x_main = np.log1p(x_raw)
        else:
            x_main = x_raw

        for follower in items:
            if follower == leader:
                continue

            y_raw = pivot.loc[follower].values.astype(float)
            if np.count_nonzero(y_raw) < min_nonzero:
                continue

            if use_log_corr:
                y_main = np.log1p(y_raw)
            else:
                y_main = y_raw

            best_lag = None
            best_corr = 0.0

            # 1) 기본 corr 기준(best_lag, best_corr) 탐색
            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue

                x_aligned = x_main[:-lag]
                y_aligned = y_main[lag:]
                corr = safe_corr(x_aligned, y_aligned)

                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            if best_lag is None:
                continue

            # 2) 스파이크 제거한 robust corr 계산
            if use_robust:
                corr_robust = robust_lagged_corr(
                    x_raw, y_raw,
                    lag=best_lag,
                    top_k=top_k_spike,
                    use_log=use_log_corr,
                )
                corr_for_threshold = corr_robust
            else:
                corr_robust = best_corr
                corr_for_threshold = best_corr

            # 3) threshold 기준 통과 여부 판단
            if abs(corr_for_threshold) < corr_threshold:
                # 스파이크 몇 개 제거하면 상관이 확 떨어진다 → 그런 쌍은 버림
                continue

            results.append({
                "leading_item_id": leader,
                "following_item_id": follower,
                "best_lag": int(best_lag),
                # max_corr는 "필터링에 실제 사용된 corr"을 쓰는 게 직관적
                "max_corr": float(corr_for_threshold),
            })

    pairs = pd.DataFrame(results)
    return pairs


def build_training_data(pivot, pairs, end_month_idx=None):
    """
    공행성쌍 + 시계열을 이용해 (X, y) 학습 데이터를 만드는 함수
    향상된 feature engineering:
    - b_t, b_t_1: 후행 품목의 현재 및 직전 달 무역량
    - a_t_lag: 선행 품목의 lag 반영된 무역량
    - max_corr, best_lag: 관계 특성
    - b_trend: 후행 품목의 추세 (최근 3개월 평균)
    - a_trend: 선행 품목의 추세 (최근 3개월 평균)
    - b_ma3: 후행 품목의 3개월 이동평균
    
    Args:
        pivot: 피벗 테이블
        pairs: 공행성 쌍 데이터프레임
        end_month_idx: 사용할 마지막 월 인덱스 (None이면 전체 사용, 검증 모드에서 사용)
    """
    months = pivot.columns.to_list()
    n_months = len(months)
    
    # end_month_idx가 지정되면 그 이전까지만 사용 (검증 모드)
    if end_month_idx is not None:
        n_months = min(n_months, end_month_idx)

    rows = []

    for row in tqdm(pairs.itertuples(index=False), total=len(pairs), desc="학습 데이터 생성"):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot.index or follower not in pivot.index:
            continue

        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)

        # t+1이 존재하고, t-lag >= 0인 구간만 학습에 사용
        for t in range(max(lag, 2), n_months - 1):
            b_t = b_series[t]
            b_t_1 = b_series[t - 1]
            a_t_lag = a_series[t - lag]
            b_t_plus_1 = b_series[t + 1]

            # 추가 feature 계산
            # 추세: 최근 3개월 평균
            if t >= 2:
                b_trend = np.mean(b_series[max(0, t-2):t+1]) if t >= 2 else b_t
                a_trend = np.mean(a_series[max(0, t-lag-2):t-lag+1]) if t-lag >= 2 else a_t_lag
            else:
                b_trend = b_t
                a_trend = a_t_lag

            # 이동평균
            if t >= 2:
                b_ma3 = np.mean(b_series[max(0, t-2):t+1])
            else:
                b_ma3 = b_t

            # 변화율
            if b_t_1 > 0:
                b_change = (b_t - b_t_1) / (b_t_1 + 1e-6)
            else:
                b_change = 0.0

            rows.append({
                "b_t": b_t,
                "b_t_1": b_t_1,
                "a_t_lag": a_t_lag,
                "max_corr": corr,
                "best_lag": float(lag),
                "b_trend": b_trend,
                "a_trend": a_trend,
                "b_ma3": b_ma3,
                "b_change": b_change,
                "target": b_t_plus_1,
            })

    df_train = pd.DataFrame(rows)
    return df_train


def predict(pivot, pairs, reg, feature_cols, predict_month_idx=None):
    """
    회귀 모델 추론 및 제출 파일 생성
    탐색된 공행성 쌍에 대해 후행 품목의 다음 달 총 무역량 예측
    
    Args:
        pivot: 피벗 테이블
        pairs: 공행성 쌍 데이터프레임
        reg: 학습된 회귀 모델
        feature_cols: feature 컬럼 리스트
        predict_month_idx: 예측할 월의 인덱스 (None이면 마지막 달 다음 달 예측, 검증 모드에서 사용)
    """
    months = pivot.columns.to_list()
    n_months = len(months)
    
    # predict_month_idx가 지정되면 해당 월을 예측 (검증 모드)
    if predict_month_idx is not None:
        t_last = predict_month_idx - 1  # 예측할 달의 이전 달
        if t_last < 0:
            return pd.DataFrame(columns=["leading_item_id", "following_item_id", "value"])
    else:
        # 가장 마지막 두 달 index (2025-7, 2025-6)
        t_last = n_months - 1
    
    t_prev = t_last - 1

    preds = []

    for row in tqdm(pairs.itertuples(index=False), total=len(pairs), desc="예측 수행"):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot.index or follower not in pivot.index:
            continue

        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)

        # t_last - lag 가 0 이상인 경우만 예측
        if t_last - lag < 0:
            continue

        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]

        # 추가 feature 계산 (build_training_data와 동일하게)
        if t_last >= 2:
            b_trend = np.mean(b_series[max(0, t_last-2):t_last+1])
            a_trend = np.mean(a_series[max(0, t_last-lag-2):t_last-lag+1]) if t_last-lag >= 2 else a_t_lag
        else:
            b_trend = b_t
            a_trend = a_t_lag

        if t_last >= 2:
            b_ma3 = np.mean(b_series[max(0, t_last-2):t_last+1])
        else:
            b_ma3 = b_t

        if b_t_1 > 0:
            b_change = (b_t - b_t_1) / (b_t_1 + 1e-6)
        else:
            b_change = 0.0

        # Feature 벡터 구성
        features = {
            "b_t": b_t,
            "b_t_1": b_t_1,
            "a_t_lag": a_t_lag,
            "max_corr": corr,
            "best_lag": float(lag),
            "b_trend": b_trend,
            "a_trend": a_trend,
            "b_ma3": b_ma3,
            "b_change": b_change,
        }

        X_test = np.array([[features[col] for col in feature_cols]])
        y_pred_log = reg.predict(X_test)[0]
        
        # 로그 변환 역변환: exp(x) - 1
        y_pred = np.expm1(y_pred_log)
        
        # 후처리: 음수 예측 → 0으로 변환, 소수점 → 정수 변환
        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred

def plot_comovement_pair(
    pivot,
    leader,
    follower,
    best_lag,
    corr=None,
    use_log=False,
    shift_follower=True,
    title_prefix="Pair",
):
    """
    한 공행성 쌍(leader, follower)을 시계열로 그려주는 함수.

    Args:
        pivot         : item_id × ym 피벗 (value 기준)
        leader        : 선행 품목 item_id
        follower      : 후행 품목 item_id
        best_lag      : 이 쌍의 best_lag (int)
        corr          : 표시용 correlation 값 (float, 옵션)
        use_log       : True면 log1p(value) 스케일로 플롯
        shift_follower: True면 follower를 lag만큼 "당겨서" leader와 align
        title_prefix  : 그래프 제목 앞에 붙일 텍스트
    """
    months = pivot.columns.to_list()
    a = pivot.loc[leader].values.astype(float)
    b = pivot.loc[follower].values.astype(float)

    if use_log:
        a_plot = np.log1p(a)
        b_plot = np.log1p(b)
        y_label = "log1p(Trade value)"
    else:
        a_plot = a
        b_plot = b
        y_label = "Trade value"

    if shift_follower:
        # leader_t vs follower_{t+lag} 를 같은 x축 월에 비교하도록 정렬
        if best_lag >= len(months):
            print(f"[경고] lag={best_lag}가 너무 커서 시각화 불가")
            return

        months_aligned = months[:-best_lag]
        leader_series = a_plot[:-best_lag]
        follower_series = b_plot[best_lag:]
    else:
        # 원 시계열 그대로 (lag 시각적 반영 X)
        months_aligned = months
        leader_series = a_plot
        follower_series = b_plot

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(months_aligned, leader_series, label=f"Leader {leader}")
    ax.plot(months_aligned, follower_series, label=f"Follower {follower}")

    title = f"{title_prefix} {leader} → {follower} (lag={best_lag}"
    if corr is not None:
        title += f", corr={corr:.3f}"
    title += ")"

    ax.set_title(title)
    ax.set_xlabel("Month (ym)")
    ax.set_ylabel(y_label)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_top_comovement_pairs(
    pivot,
    pairs,
    top_k=5,
    use_log=False,
    shift_follower=True,
):
    """
    공행성 쌍 DataFrame에서 |max_corr| 상위 top_k 개를 골라
    각 쌍을 plot_comovement_pair로 차례대로 그려주는 함수.

    Args:
        pivot        : item_id × ym 피벗 (value 기준)
        pairs        : find_comovement_pairs 결과 DataFrame
        top_k        : 그릴 상위 쌍 개수
        use_log      : True면 log1p 스케일로 플롯
        shift_follower: True면 follower를 lag만큼 당겨서 leader와 align
    """
    if len(pairs) == 0:
        print("pairs 가 비어 있습니다.")
        return

    # |max_corr| 기준으로 내림차순 정렬 후 상위 top_k
    pairs_sorted = pairs.reindex(
        pairs["max_corr"].abs().sort_values(ascending=False).index
    ).head(top_k)

    for idx, row in pairs_sorted.iterrows():
        leader = row["leading_item_id"]
        follower = row["following_item_id"]
        lag = int(row["best_lag"])
        corr = float(row["max_corr"])

        plot_comovement_pair(
            pivot,
            leader,
            follower,
            best_lag=lag,
            corr=corr,
            use_log=use_log,
            shift_follower=shift_follower,
            title_prefix="Top pair",
        )


def cross_validate_with_kfold(pivot, pairs, feature_cols, is_validate_mode=False, 
                               answer_df=None, n_splits=5):
    """
    KFold 교차 검증을 수행하고 모든 fold의 예측을 앙상블하는 함수
    
    Args:
        pivot: 피벗 테이블
        pairs: 공행성 쌍 데이터프레임
        feature_cols: feature 컬럼 리스트
        is_validate_mode: 검증 모드 여부
        answer_df: 검증 모드일 때 정답 데이터프레임 (None이면 평가 안 함)
        n_splits: KFold 분할 수 (기본값: 5)
    
    Returns:
        final_submission: 모든 fold의 예측을 앙상블한 최종 예측 결과
        fold_scores: 각 fold의 점수 리스트 (validate 모드인 경우)
    """
    # 학습 데이터 생성
    print("\n[KFold] 학습 데이터 생성 중...")
    df_train_model = build_training_data(pivot, pairs)
    print(f'생성된 학습 데이터 shape: {df_train_model.shape}')
    
    if len(df_train_model) == 0:
        print("경고: 학습 데이터가 없습니다.")
        return pd.DataFrame(columns=["leading_item_id", "following_item_id", "value"]), []
    
    train_X = df_train_model[feature_cols].values
    train_y = df_train_model["target"].values
    
    # KFold 생성
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 모든 fold의 예측 결과를 저장할 딕셔너리
    all_predictions = {}  # {(leading_item_id, following_item_id): [pred1, pred2, ...]}
    fold_scores = []
    
    print(f"\n[KFold] {n_splits}-Fold 교차 검증 시작...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_X), 1):
        print(f"\n--- Fold {fold_idx}/{n_splits} ---")
        
        # 학습/검증 데이터 분할
        X_train_fold = train_X[train_idx]
        y_train_fold = train_y[train_idx]
        X_val_fold = train_X[val_idx]
        y_val_fold = train_y[val_idx]
        
        print(f"  학습 샘플 수: {len(X_train_fold)}, 검증 샘플 수: {len(X_val_fold)}")
        
        # 모델 학습 (과적합 방지를 위한 보수적 튜닝)
        reg = xgb.XGBRegressor(
            n_estimators=250,  # 트리 개수 약간 증가
            max_depth=6,  # 깊이 유지
            learning_rate=0.08,  # 학습률 약간 감소 (더 안정적인 학습)
            subsample=0.85,  # 샘플 비율 약간 증가
            colsample_bytree=0.85,  # Feature 샘플 비율 약간 증가
            min_child_weight=3,  # 과적합 방지
            reg_alpha=0.05,  # L1 정규화 (약한 정규화)
            reg_lambda=0.5,  # L2 정규화 (약한 정규화)
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        # 로그 변환된 타겟으로 학습
        y_train_fold_log = np.log1p(y_train_fold)
        reg.fit(X_train_fold, y_train_fold_log)
        
        # 전체 pairs에 대해 예측 (실제 제출 형식)
        # 각 fold에서 학습된 모델로 모든 공행성 쌍에 대해 예측
        fold_submission = predict(pivot, pairs, reg, feature_cols)
        
        # 예측 결과를 딕셔너리에 누적
        for _, pred_row in fold_submission.iterrows():
            pair_key = (pred_row['leading_item_id'], pred_row['following_item_id'])
            if pair_key not in all_predictions:
                all_predictions[pair_key] = []
            all_predictions[pair_key].append(pred_row['value'])
        
        # validate 모드인 경우 이 fold의 점수 계산
        if is_validate_mode and answer_df is not None:
            try:
                score = comovement_score(answer_df, fold_submission)
                f1 = comovement_f1(answer_df, fold_submission)
                nmae = comovement_nmae(answer_df, fold_submission)
                
                fold_scores.append({
                    'fold': fold_idx,
                    'f1': f1,
                    'nmae': nmae,
                    'score': score
                })
                
                print(f"  Fold {fold_idx} 점수: F1={f1:.6f}, NMAE={nmae:.6f}, Score={score:.6f}")
            except Exception as e:
                print(f"  Fold {fold_idx} 평가 중 오류: {e}")
    
    # 모든 fold의 예측을 앙상블하여 최종 예측 생성
    print("\n[KFold] 모든 fold의 예측을 앙상블 중...")
    final_rows = []
    for pair_key, predictions in all_predictions.items():
        # 중앙값 사용 (이상치에 더 robust)
        # 평균과 중앙값의 평균 사용 (안정성과 정확도 균형)
        median_pred = np.median(predictions)
        mean_pred = np.mean(predictions)
        # 가중 평균: 중앙값 60%, 평균 40%
        avg_pred = 0.6 * median_pred + 0.4 * mean_pred
        # 후처리: 음수는 0으로, 소수점은 정수로 반올림
        avg_pred = max(0.0, float(avg_pred))
        avg_pred = int(round(avg_pred))
        
        final_rows.append({
            'leading_item_id': pair_key[0],
            'following_item_id': pair_key[1],
            'value': avg_pred
        })
    
    final_submission = pd.DataFrame(final_rows)
    
    # validate 모드인 경우 평균 점수 출력
    if is_validate_mode and len(fold_scores) > 0:
        avg_f1 = np.mean([s['f1'] for s in fold_scores])
        avg_nmae = np.mean([s['nmae'] for s in fold_scores])
        avg_score = np.mean([s['score'] for s in fold_scores])
        
        print("\n" + "=" * 60)
        print("KFold 교차 검증 결과 (평균)")
        print("=" * 60)
        print(f"평균 F1 Score: {avg_f1:.6f}")
        print(f"평균 NMAE: {avg_nmae:.6f}")
        print(f"평균 Final Score: {avg_score:.6f}")
        print(f"  (Score = 0.6 × F1 + 0.4 × (1 - NMAE))")
        print("=" * 60)
    
    return final_submission, fold_scores


def main():
    """메인 실행 함수"""
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='국민대학교 AI빅데이터 분석 경진대회 - 학습 및 예측')
    parser.add_argument('--mode', type=str, default='submit', choices=['submit', 'validate'],
                        help='실행 모드: submit(전체 학습 및 제출 파일 생성, 기본값) 또는 validate(검증 모드)')
    args = parser.parse_args()
    
    is_validate_mode = (args.mode == 'validate')
    
    print("=" * 60)
    if is_validate_mode:
        print("국민대학교 AI빅데이터 분석 경진대회 - 검증 모드")
    else:
        print("국민대학교 AI빅데이터 분석 경진대회 - 학습 및 예측")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1단계] 데이터 로드 중...")
    train = pd.read_csv('../data/train.csv')
    print(f"학습 데이터 shape: {train.shape}")

    # 2. 데이터 전처리
    print("\n[2단계] 데이터 전처리 중...")
    # year, month, item_id 기준으로 value 합산
    monthly = (
        train
        .groupby(["item_id", "year", "month"], as_index=False)["value"]
        .sum()
    )

    # year, month를 하나의 키(ym)로 묶기
    monthly["ym"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )

    # item_id × ym 피벗 (월별 총 무역량 매트릭스 생성)
    pivot = (
        monthly
        .pivot(index="item_id", columns="ym", values="value")
        .fillna(0.0)
    )
    print(f"피벗 테이블 shape: {pivot.shape}")
    
    months = pivot.columns.to_list()
    n_months = len(months)
    
    # 검증 모드일 때 데이터 분할 (9:1)
    if is_validate_mode:
        # 시계열 데이터이므로 시간 순서대로 분할 (마지막 10%를 검증 데이터로)
        split_idx = int(n_months * 0.9)
        train_months = months[:split_idx]
        val_months = months[split_idx:]
        
        print(f"\n[검증 모드] 데이터 분할:")
        print(f"  학습 기간: {train_months[0]} ~ {train_months[-1]} ({len(train_months)}개월)")
        print(f"  검증 기간: {val_months[0]} ~ {val_months[-1]} ({len(val_months)}개월)")
        
        # 학습용 피벗 테이블 (검증 기간 제외)
        pivot_train = pivot[train_months].copy()
        
        # 검증 기간 전체를 포함한 피벗 테이블 (정답 공행성 쌍 탐색용)
        # 검증 기간의 마지막 달까지 포함하여 공행성 쌍을 탐색
        # 이렇게 하면 검증 기간의 데이터도 활용하여 더 정확한 공행성 쌍을 찾을 수 있음
        pivot_for_answer = pivot[months[:split_idx + len(val_months)]].copy()
        
        # 검증 기간의 첫 번째 달이 예측 대상
        predict_month_idx = split_idx  # 검증 기간의 첫 번째 달 인덱스
    else:
        pivot_train = pivot.copy()
        pivot_for_answer = None
        predict_month_idx = None

    # 3. 공행성쌍 탐색
    print("\n[3단계] 공행성 쌍 탐색 중...")
    if is_validate_mode:
        # 검증 모드: 학습 데이터로 예측할 공행성 쌍 탐색
        pairs = find_comovement_pairs(
            pivot_train, 
            max_lag=6, 
            min_nonzero=12, 
            corr_threshold=0.35,
            use_log_corr=True,
            use_robust=True,
            top_k_spike=2
            )
        print(f"학습 데이터로 탐색된 공행성쌍 수: {len(pairs)}")

        # 공행성 시각화 (상위 5개, lag 반영 + 로그 스케일)
        print("\n[디버그] 상위 공행성쌍 5개 시각화 중...")
        plot_top_comovement_pairs(
            pivot,
            pairs,
            top_k=5,
            use_log=True,         # log1p(value)로 보기
            shift_follower=True,  # lag 반영해서 follower를 당겨서 보기
        )
        
        # 정답 공행성 쌍 탐색 (검증 기간 전체 포함)
        # 실제 대회에서는 정답 공행성 쌍이 미리 정해져 있지만,
        # 검증 모드에서는 검증 기간 전체 데이터를 사용하여 더 정확한 공행성 쌍을 찾음
        answer_pairs = find_comovement_pairs(pivot_for_answer, max_lag=6, min_nonzero=12, corr_threshold=0.35)
        print(f"정답 공행성쌍 수 (검증 기간 포함): {len(answer_pairs)}")
    else:
        pairs = find_comovement_pairs(pivot_train, max_lag=6, min_nonzero=12, corr_threshold=0.35)
        print(f"탐색된 공행성쌍 수: {len(pairs)}")

    if len(pairs) == 0:
        print("경고: 공행성 쌍이 발견되지 않았습니다.")
        return

    # 4. Feature 컬럼 정의
    feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                    'b_trend', 'a_trend', 'b_ma3', 'b_change']

    # 5. KFold 교차 검증 수행
    print("\n[5단계] KFold 교차 검증 수행 중...")
    
    if is_validate_mode:
        # 검증용 정답 데이터 생성
        # 정답 공행성 쌍(answer_pairs)에 대해서만 정답 생성
        # 검증 기간의 첫 번째 달(predict_month_idx)의 실제 무역량을 정답으로 사용
        answer_rows = []
        for _, row in answer_pairs.iterrows():
            following_item_id = row['following_item_id']
            if following_item_id in pivot.index and predict_month_idx < n_months:
                # 실제 검증 기간의 첫 번째 달 무역량
                actual_value = pivot.loc[following_item_id, months[predict_month_idx]]
                answer_rows.append({
                    'leading_item_id': row['leading_item_id'],
                    'following_item_id': following_item_id,
                    'value': int(round(float(actual_value)))  # 정수로 변환
                })
        
        answer_df = pd.DataFrame(answer_rows)
        print(f"정답 쌍 수: {len(answer_df)}")
        
        # KFold 교차 검증 수행 (학습 데이터만 사용)
        submission, fold_scores = cross_validate_with_kfold(
            pivot_train, pairs, feature_cols, 
            is_validate_mode=True, answer_df=answer_df, n_splits=5
        )
        
        print(f"\n예측된 쌍 수: {len(submission)}")
        
        # 예측 쌍과 정답 쌍의 교집합 확인
        pred_pairs = set(zip(submission['leading_item_id'], submission['following_item_id']))
        ans_pairs = set(zip(answer_df['leading_item_id'], answer_df['following_item_id']))
        intersection = pred_pairs & ans_pairs
        print(f"일치하는 쌍 수: {len(intersection)}")
        
        # 최종 앙상블 결과 평가
        print("\n[6단계] 최종 앙상블 결과 평가 중...")
        if len(answer_df) > 0 and len(submission) > 0:
            try:
                score = comovement_score(answer_df, submission)
                f1 = comovement_f1(answer_df, submission)
                nmae = comovement_nmae(answer_df, submission)
                
                # 상세 분석
                pred_pairs = set(zip(submission['leading_item_id'], submission['following_item_id']))
                ans_pairs = set(zip(answer_df['leading_item_id'], answer_df['following_item_id']))
                tp = len(pred_pairs & ans_pairs)
                fp = len(pred_pairs - ans_pairs)
                fn = len(ans_pairs - pred_pairs)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                print("\n" + "=" * 60)
                print("최종 앙상블 검증 결과")
                print("=" * 60)
                print(f"공행성 쌍 판별:")
                print(f"  Precision: {precision:.6f}")
                print(f"  Recall: {recall:.6f}")
                print(f"  F1 Score: {f1:.6f}")
                print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
                print(f"\n무역량 예측:")
                print(f"  NMAE: {nmae:.6f}")
                print(f"\n최종 점수:")
                print(f"  Final Score: {score:.6f}")
                print(f"  (Score = 0.6 × F1 + 0.4 × (1 - NMAE))")
                print("=" * 60)
            except Exception as e:
                print(f"평가 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                print("예측 결과와 정답 데이터를 확인해주세요.")
        else:
            print("경고: 평가할 데이터가 없습니다.")
            if len(answer_df) == 0:
                print("  - 정답 데이터가 생성되지 않았습니다.")
            if len(submission) == 0:
                print("  - 예측 데이터가 생성되지 않았습니다.")
    else:
        # 제출 모드: 전체 데이터로 KFold 교차 검증 수행
        submission, _ = cross_validate_with_kfold(
            pivot_train, pairs, feature_cols, 
            is_validate_mode=False, answer_df=None, n_splits=5
        )
        print(f"\n예측된 쌍 수: {len(submission)}")

        # 6. 제출 파일 저장
        print("\n[6단계] 제출 파일 저장 중...")
        output_path = '../data/submission.csv'
        submission.to_csv(output_path, index=False)
        print(f"제출 파일 저장 완료: {output_path}")
        print(f"\n제출 파일 미리보기:")
        print(submission.head(10))
        print(f"\n총 {len(submission)}개의 공행성 쌍에 대한 예측이 완료되었습니다.")


if __name__ == "__main__":
    main()

