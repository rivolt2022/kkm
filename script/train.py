"""
국민대학교 AI빅데이터 분석 경진대회 - 공행성 쌍 판별 및 무역량 예측
앙상블 버전: XGBoost, LightGBM, CatBoost 세 가지 모델 앙상블 및 향상된 feature engineering 사용
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tqdm import tqdm
import warnings
import argparse
import sys
import os
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')

# evaluation.py import를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../document'))
from evaluation import comovement_score, comovement_f1, comovement_nmae


def safe_corr(x, y):
    """안전한 상관계수 계산 (표준편차가 0인 경우 처리)"""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_item_statistics(pivot):
    """
    품목별 통계량 계산 (research3.txt의 groupby 통계량 아이디어)
    품목별 mean, std, cv, zero_ratio 등을 계산하여 딕셔너리로 반환
    """
    item_stats = {}
    
    for item_id in pivot.index:
        series = pivot.loc[item_id].values.astype(float)
        non_zero = series[series > 0]
        
        mean_val = np.mean(series)
        std_val = np.std(series)
        
        item_stats[item_id] = {
            'mean': mean_val,
            'std': std_val,
            'cv': std_val / (mean_val + 1e-6),  # 변동계수
            'zero_ratio': (len(series) - len(non_zero)) / len(series),
            'nonzero_count': len(non_zero),
        }
    
    return item_stats


def find_comovement_pairs(pivot, max_lag=6, min_nonzero=12, corr_threshold=0.35):
    """
    공행성 쌍 탐색 (시간 윈도우별 독립 탐색 후 통합)
    
    전략:
    1. 전체 기간을 여러 윈도우로 나눔 (각 18개월, 50% 오버랩)
    2. 각 윈도우에서 독립적으로 공행성 쌍 탐색
    3. 여러 윈도우에서 일관되게 나타나는 쌍만 최종 선택
    4. 최종 상관계수는 전체 기간 기준으로 계산
    
    이 방법의 장점:
    - 시간에 따라 변화하는 패턴도 포착 가능
    - 일시적인 노이즈 관계는 필터링됨
    - 일관된 관계만 선택하여 안정성 향상
    """
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)
    
    # 윈도우 설정: 각 18개월, 50% 오버랩
    window_size = 18
    window_step = window_size // 2  # 50% 오버랩
    
    # 윈도우별로 탐색
    window_results = {}  # {(leader, follower): [윈도우별 결과]}
    
    print(f"[1단계] 시간 윈도우별 탐색 (윈도우 크기: {window_size}개월, 오버랩: 50%)...")
    
    for window_start in range(0, n_months - window_size + 1, window_step):
        window_end = min(window_start + window_size, n_months)
        window_months = months[window_start:window_end]
        window_pivot = pivot[window_months].copy()
        
        print(f"  윈도우 {window_start//window_step + 1}: {window_months[0]} ~ {window_months[-1]}")
        
        # 이 윈도우에서 공행성 쌍 탐색
        for i, leader in enumerate(items):
            x = window_pivot.loc[leader].values.astype(float) if leader in window_pivot.index else np.zeros(len(window_months))
            if np.count_nonzero(x) < min_nonzero // 2:  # 윈도우가 작으므로 임계값 낮춤
                continue

            for follower in items:
                if follower == leader:
                    continue

                y = window_pivot.loc[follower].values.astype(float) if follower in window_pivot.index else np.zeros(len(window_months))
                if np.count_nonzero(y) < min_nonzero // 2:
                    continue

                best_lag = None
                best_corr = 0.0
                window_n = len(window_months)

                # lag = 1 ~ max_lag 탐색
                for lag in range(1, min(max_lag + 1, window_n)):
                    if window_n <= lag:
                        continue
                    corr = safe_corr(x[:-lag], y[lag:])
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag

                # 임계값 이상이면 이 윈도우에서 발견된 것으로 기록
                if best_lag is not None and abs(best_corr) >= corr_threshold:
                    pair_key = (leader, follower)
                    if pair_key not in window_results:
                        window_results[pair_key] = []
                    window_results[pair_key].append({
                        'lag': best_lag,
                        'corr': best_corr,
                        'window': window_start // window_step
                    })
    
    print(f"[2단계] 윈도우 결과 통합 및 최종 평가...")
    print(f"  발견된 후보 쌍 수: {len(window_results)}")
    
    # 최소 지지도: 최소 2개 윈도우에서 나타나야 함
    min_support = max(2, (n_months // window_step) // 3)  # 전체 윈도우의 1/3 이상
    
    results = []
    
    for (leader, follower), window_data in tqdm(window_results.items(), desc="최종 평가"):
        # 최소 지지도 확인
        if len(window_data) < min_support:
            continue
        
        # 전체 기간에서 최종 상관계수 계산
        x = pivot.loc[leader].values.astype(float)
        y = pivot.loc[follower].values.astype(float)
        
        best_lag = None
        best_corr = 0.0
        
        # 전체 기간에서 최적 lag와 상관계수 찾기
        for lag in range(1, max_lag + 1):
            if n_months <= lag:
                continue
            corr = safe_corr(x[:-lag], y[lag:])
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        
        # 최종 임계값 확인 및 채택
        if best_lag is not None and abs(best_corr) >= corr_threshold:
            results.append({
                "leading_item_id": leader,
                "following_item_id": follower,
                "best_lag": best_lag,
                "max_corr": best_corr,
            })
    
    pairs = pd.DataFrame(results)
    print(f"최종 채택된 쌍 수: {len(pairs)} (최소 {min_support}개 윈도우에서 발견)")
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
    - b_change: 지난 달 대비 변화율
    
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
        
        # 후처리: 음수 예측 → 0으로 변환
        y_pred = max(0.0, float(y_pred))
        
        # 합리적인 범위로 클리핑 (b_t 기준 0.1배 ~ 10배 범위로 제한)
        if b_t > 0:
            y_pred = np.clip(y_pred, b_t * 0.1, b_t * 10.0)
        
        # 소수점 → 정수 변환
        y_pred = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred


def cross_validate_with_kfold(pivot, pairs, feature_cols, is_validate_mode=False, 
                               answer_df=None, n_splits=5, seeds=[42, 1031, 106],
                               model_weights=None):
    """
    KFold 교차 검증을 수행하고 모든 fold의 예측을 앙상블하는 함수
    XGBoost, LightGBM, CatBoost 세 가지 모델을 모두 사용하여 앙상블
    research3.txt의 Seed Ensemble 전략 적용: 여러 seed로 학습하여 다양성 증가
    
    Args:
        pivot: 피벗 테이블
        pairs: 공행성 쌍 데이터프레임
        feature_cols: feature 컬럼 리스트
        is_validate_mode: 검증 모드 여부
        answer_df: 검증 모드일 때 정답 데이터프레임 (None이면 평가 안 함)
        n_splits: KFold 분할 수 (기본값: 5)
        seeds: 사용할 seed 리스트 (기본값: [42, 1031, 106], research3.txt 참고)
        model_weights: 모델별 가중치 딕셔너리 (None이면 기본값 사용)
    
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
    
    # Seed Ensemble: 여러 seed로 학습하여 다양성 증가
    print(f"\n[Seed Ensemble] {len(seeds)}개 seed로 학습 시작 (research3.txt 전략)")
    all_seed_predictions = {}  # {seed: {model_name: {pair_key: [preds]}}}
    all_fold_scores = []
    
    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}")
        print(f"[Seed {seed_idx}/{len(seeds)}] Seed={seed} 학습 시작")
        print(f"{'='*60}")
        
        # KFold 생성 (seed별로 다른 분할)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
        # 모델별 예측 결과를 저장할 딕셔너리 (모델별로 구분)
        model_predictions = {}  # {model_name: {(leading_item_id, following_item_id): [pred1, pred2, ...]}}
        fold_scores = []
        
        # 사용할 모델 리스트
        models = [
            ('XGBoost', 'xgb'),
            ('LightGBM', 'lgb'),
            ('CatBoost', 'cb')
        ]
        
        print(f"\n[Seed {seed}] {len(models)}개 모델 × {n_splits}-Fold 교차 검증 시작...")
        
        # 각 모델마다 KFold 수행
        for model_name, model_type in models:
            print(f"\n--- [{model_name}] {n_splits}-Fold 교차 검증 시작 ---")
            
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_X), 1):
                print(f"\n--- {model_name} Fold {fold_idx}/{n_splits} ---")
                
                # 학습/검증 데이터 분할
                X_train_fold = train_X[train_idx]
                y_train_fold = train_y[train_idx]
                X_val_fold = train_X[val_idx]
                y_val_fold = train_y[val_idx]
                
                print(f"  학습 샘플 수: {len(X_train_fold)}, 검증 샘플 수: {len(X_val_fold)}")
                
                # 로그 변환된 타겟으로 학습
                y_train_fold_log = np.log1p(y_train_fold)
                
                # 모델별 학습 (seed 적용)
                # 하이퍼파라미터는 optimize_hyperparameters_grid.py 실행 결과 반영
                if model_type == 'xgb':
                    reg = xgb.XGBRegressor(
                        n_estimators=250,
                        max_depth=7,  # 최적화: 6 -> 7
                        learning_rate=0.09,  # 최적화: 0.08 -> 0.09
                        subsample=0.8,  # 최적화: 0.85 -> 0.8
                        colsample_bytree=0.9,  # 최적화: 0.85 -> 0.9
                        min_child_weight=3,
                        reg_alpha=0.05,
                        reg_lambda=0.5,
                        random_state=seed,  # seed 적용
                        n_jobs=-1,
                        verbosity=0
                    )
                elif model_type == 'lgb':
                    reg = lgb.LGBMRegressor(
                        n_estimators=250,
                        max_depth=7,  # 최적화: 6 -> 7
                        learning_rate=0.07,  # 최적화: 0.08 -> 0.07
                        subsample=0.8,  # 최적화: 0.85 -> 0.8
                        colsample_bytree=0.85,
                        min_child_samples=3,
                        reg_alpha=0.03,  # 최적화: 0.05 -> 0.03
                        reg_lambda=0.5,
                        random_state=seed,  # seed 적용
                        n_jobs=-1,
                        verbosity=-1
                    )
                elif model_type == 'cb':
                    reg = cb.CatBoostRegressor(
                        iterations=250,
                        depth=7,  # 최적화: 6 -> 7
                        learning_rate=0.09,  # 최적화: 0.08 -> 0.09
                        subsample=0.8,  # 최적화: 0.85 -> 0.8
                        colsample_bylevel=0.9,  # 최적화: 0.85 -> 0.9
                        min_data_in_leaf=3,
                        l2_leaf_reg=0.6,  # 최적화: 0.5 -> 0.6
                        random_state=seed,  # seed 적용
                        thread_count=-1,
                        verbose=False
                    )
                
                reg.fit(X_train_fold, y_train_fold_log)
                
                # 전체 pairs에 대해 예측 (실제 제출 형식)
                fold_submission = predict(pivot, pairs, reg, feature_cols)
                
                # 예측 결과를 모델별로 딕셔너리에 누적
                if model_name not in model_predictions:
                    model_predictions[model_name] = {}
                for _, pred_row in fold_submission.iterrows():
                    pair_key = (pred_row['leading_item_id'], pred_row['following_item_id'])
                    if pair_key not in model_predictions[model_name]:
                        model_predictions[model_name][pair_key] = []
                    model_predictions[model_name][pair_key].append(pred_row['value'])
                
                # validate 모드인 경우 이 fold의 점수 계산
                if is_validate_mode and answer_df is not None:
                    try:
                        score = comovement_score(answer_df, fold_submission)
                        f1 = comovement_f1(answer_df, fold_submission)
                        nmae = comovement_nmae(answer_df, fold_submission)
                        
                        fold_scores.append({
                            'model': model_name,
                            'fold': fold_idx,
                            'f1': f1,
                            'nmae': nmae,
                            'score': score
                        })
                        
                        print(f"  {model_name} Fold {fold_idx} 점수: F1={f1:.6f}, NMAE={nmae:.6f}, Score={score:.6f}")
                    except Exception as e:
                        print(f"  {model_name} Fold {fold_idx} 평가 중 오류: {e}")
        
        # 이 seed의 예측 결과 저장
        all_seed_predictions[seed] = model_predictions
        all_fold_scores.extend(fold_scores)
    
    # Seed Ensemble: 모든 seed의 예측을 평균하여 최종 앙상블
    print("\n" + "="*60)
    print("[Seed Ensemble] 모든 seed의 예측을 평균하여 최종 앙상블 중...")
    print("="*60)
    
    # 모델별 가중치 (optimize_models_independently.py 실행 결과 반영)
    # 최적 가중치: XGBoost=0.40, LightGBM=0.30, CatBoost=0.30 (Score: 0.632840)
    if model_weights is None:
        model_weights = {'XGBoost': 0.40, 'LightGBM': 0.30, 'CatBoost': 0.30}
    
    # 모든 pair_key 수집
    all_pairs = set()
    for seed in all_seed_predictions:
        for model_name in all_seed_predictions[seed]:
            all_pairs.update(all_seed_predictions[seed][model_name].keys())
    
    final_rows = []
    for pair_key in all_pairs:
        weighted_sum = 0.0
        total_weight = 0.0
        
        # 각 모델의 모든 seed × 모든 fold 예측에 가중치 적용
        for model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
            model_all_preds = []  # 이 모델의 모든 seed × 모든 fold 예측
            
            # 모든 seed에서 이 모델의 예측 수집
            for seed in all_seed_predictions:
                if model_name in all_seed_predictions[seed] and pair_key in all_seed_predictions[seed][model_name]:
                    model_all_preds.extend(all_seed_predictions[seed][model_name][pair_key])
            
            if len(model_all_preds) > 0:
                model_avg = np.mean(model_all_preds)  # 모든 seed × 모든 fold의 평균
                weight = model_weights[model_name]
                weighted_sum += model_avg * weight
                total_weight += weight
        
        # 가중 평균 계산
        avg_pred = weighted_sum / total_weight if total_weight > 0 else 0.0
        
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
    if is_validate_mode and len(all_fold_scores) > 0:
        # 모델별 평균 점수 (모든 seed 포함)
        print("\n" + "=" * 60)
        print("모델별 KFold 교차 검증 결과 (평균, 모든 seed 포함)")
        print("=" * 60)
        for model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
            model_scores = [s for s in all_fold_scores if s['model'] == model_name]
            if len(model_scores) > 0:
                avg_f1 = np.mean([s['f1'] for s in model_scores])
                avg_nmae = np.mean([s['nmae'] for s in model_scores])
                avg_score = np.mean([s['score'] for s in model_scores])
                print(f"{model_name}:")
                print(f"  평균 F1 Score: {avg_f1:.6f}")
                print(f"  평균 NMAE: {avg_nmae:.6f}")
                print(f"  평균 Final Score: {avg_score:.6f}")
        
        # 전체 평균 점수 (모든 seed × 모든 모델 × 모든 fold)
        avg_f1 = np.mean([s['f1'] for s in all_fold_scores])
        avg_nmae = np.mean([s['nmae'] for s in all_fold_scores])
        avg_score = np.mean([s['score'] for s in all_fold_scores])
        
        print("\n" + "=" * 60)
        print("전체 평균 (모든 모델 × 모든 fold)")
        print("=" * 60)
        print(f"평균 F1 Score: {avg_f1:.6f}")
        print(f"평균 NMAE: {avg_nmae:.6f}")
        print(f"평균 Final Score: {avg_score:.6f}")
        print(f"  (Score = 0.6 × F1 + 0.4 × (1 - NMAE))")
        print("=" * 60)
    
    return final_submission, all_fold_scores


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
    # F1 Score 향상을 위한 보수적인 파라미터 조정:
    # - corr_threshold: 0.35 → 0.34 (약간만 낮춰서 과적합 방지)
    # - min_nonzero: 12 유지 (데이터 품질 유지)
    corr_threshold = 0.34
    min_nonzero = 12
    
    if is_validate_mode:
        # 검증 모드: 학습 데이터로 예측할 공행성 쌍 탐색
        pairs = find_comovement_pairs(pivot_train, max_lag=6, min_nonzero=min_nonzero, corr_threshold=corr_threshold)
        print(f"학습 데이터로 탐색된 공행성쌍 수: {len(pairs)}")
        
        # 정답 공행성 쌍 탐색 (검증 기간 전체 포함)
        # 실제 대회에서는 정답 공행성 쌍이 미리 정해져 있지만,
        # 검증 모드에서는 검증 기간 전체 데이터를 사용하여 더 정확한 공행성 쌍을 찾음
        answer_pairs = find_comovement_pairs(pivot_for_answer, max_lag=6, min_nonzero=min_nonzero, corr_threshold=corr_threshold)
        print(f"정답 공행성쌍 수 (검증 기간 포함): {len(answer_pairs)}")
    else:
        pairs = find_comovement_pairs(pivot_train, max_lag=6, min_nonzero=min_nonzero, corr_threshold=corr_threshold)
        print(f"탐색된 공행성쌍 수: {len(pairs)}")

    if len(pairs) == 0:
        print("경고: 공행성 쌍이 발견되지 않았습니다.")
        return

    # 4. Feature 컬럼 정의
    # 분석 결과: 기존 9개 피처가 가장 좋은 성능을 보임
    # 새로운 피처들은 높은 상관관계(중복) 또는 낮은 중요도로 인해 성능 하락
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

