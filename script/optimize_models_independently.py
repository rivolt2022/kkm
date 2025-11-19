"""
단일 모델 최적화 후 앙상블
각 모델을 독립적으로 최적화한 후 앙상블 성능 검증
"""

import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm

# evaluation.py import를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../document'))
from evaluation import comovement_score, comovement_f1, comovement_nmae

from train import (
    find_comovement_pairs, build_training_data, predict
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import KFold


def optimize_and_ensemble(pivot_train, pivot_for_answer, feature_cols, n_splits=5, seeds=[42]):
    """
    각 모델을 독립적으로 최적화한 후 앙상블
    train.py의 현재 설정에 맞춰 하이퍼파라미터 및 예측 후처리 적용
    """
    print("=" * 60)
    print("단일 모델 최적화 후 앙상블")
    print("=" * 60)
    
    # 공행성 쌍 탐색 (train.py와 동일한 파라미터)
    corr_threshold = 0.34
    min_nonzero = 12
    pairs = find_comovement_pairs(pivot_train, max_lag=6, min_nonzero=min_nonzero, corr_threshold=corr_threshold)
    answer_pairs = find_comovement_pairs(pivot_for_answer, max_lag=6, min_nonzero=min_nonzero, corr_threshold=corr_threshold)
    
    # 정답 데이터 생성
    months = pivot_for_answer.columns.to_list()
    n_months = len(months)
    split_idx = int(n_months * 0.9)
    predict_month_idx = split_idx
    
    answer_rows = []
    for _, row in answer_pairs.iterrows():
        following_item_id = row['following_item_id']
        if following_item_id in pivot_for_answer.index and predict_month_idx < n_months:
            actual_value = pivot_for_answer.loc[following_item_id, months[predict_month_idx]]
            answer_rows.append({
                'leading_item_id': row['leading_item_id'],
                'following_item_id': following_item_id,
                'value': int(round(float(actual_value)))
            })
    answer_df = pd.DataFrame(answer_rows)
    
    # 학습 데이터 생성
    df_train = build_training_data(pivot_train, pairs)
    X = df_train[feature_cols].values
    y = df_train["target"].values
    y_log = np.log1p(y)
    
    # train.py의 현재 하이퍼파라미터 사용 (optimize_hyperparameters_grid.py 최적화 결과 반영)
    print("\n[1단계] 각 모델을 독립적으로 학습 및 예측...")
    print(f"KFold n_splits={n_splits}, Seeds={seeds}")
    
    # Seed Ensemble: 여러 seed로 학습하여 다양성 증가 (train.py와 동일한 전략)
    all_seed_predictions = {}  # {seed: {model_name: {pair_key: [preds]}}}
    
    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}")
        print(f"[Seed {seed_idx}/{len(seeds)}] Seed={seed} 학습 시작")
        print(f"{'='*60}")
        
        # KFold 생성 (seed별로 다른 분할)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        model_predictions = {}  # {model_name: {pair_key: [preds]}}
        
        # 사용할 모델 리스트
        models = [
            ('XGBoost', 'xgb'),
            ('LightGBM', 'lgb'),
            ('CatBoost', 'cb')
        ]
        
        # 각 모델마다 KFold 수행
        for model_name, model_type in models:
            print(f"\n--- [{model_name}] {n_splits}-Fold 교차 검증 시작 ---")
            
            if model_name not in model_predictions:
                model_predictions[model_name] = {}
            
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
                print(f"  Fold {fold_idx}/{n_splits}")
                
                X_train_fold = X[train_idx]
                y_train_fold = y_log[train_idx]
                
                # 모델별 학습 (train.py와 동일한 하이퍼파라미터)
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
                
                reg.fit(X_train_fold, y_train_fold)
                
                # 예측 (predict 함수가 자동으로 클리핑 등 후처리 수행)
                fold_submission = predict(pivot_train, pairs, reg, feature_cols)
                
                # 예측 결과 누적
                for _, pred_row in fold_submission.iterrows():
                    pair_key = (pred_row['leading_item_id'], pred_row['following_item_id'])
                    if pair_key not in model_predictions[model_name]:
                        model_predictions[model_name][pair_key] = []
                    model_predictions[model_name][pair_key].append(pred_row['value'])
        
        # 이 seed의 예측 결과 저장
        all_seed_predictions[seed] = model_predictions
    
    # 모든 seed의 예측을 평균하여 모델별 최종 예측 생성
    print("\n[Seed Ensemble] 모든 seed의 예측을 평균하여 모델별 최종 예측 생성 중...")
    final_model_predictions = {}  # {model_name: {pair_key: [preds]}}
    
    for model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
        final_model_predictions[model_name] = {}
        all_pairs_for_model = set()
        
        # 모든 seed에서 이 모델의 pair_key 수집
        for seed in all_seed_predictions:
            if model_name in all_seed_predictions[seed]:
                all_pairs_for_model.update(all_seed_predictions[seed][model_name].keys())
        
        # 모든 seed의 예측을 평균
        for pair_key in all_pairs_for_model:
            all_preds = []
            for seed in all_seed_predictions:
                if model_name in all_seed_predictions[seed] and pair_key in all_seed_predictions[seed][model_name]:
                    all_preds.extend(all_seed_predictions[seed][model_name][pair_key])
            
            if len(all_preds) > 0:
                final_model_predictions[model_name][pair_key] = all_preds
    
    # final_model_predictions를 model_predictions로 사용
    model_predictions = final_model_predictions
    
    # 각 모델의 단독 성능 평가
    print("\n[2단계] 각 모델의 단독 성능 평가...")
    model_scores = {}
    
    for model_name in model_predictions:
        # 모델별 평균 예측
        model_submission_rows = []
        for pair_key, preds in model_predictions[model_name].items():
            avg_pred = int(round(np.mean(preds)))
            model_submission_rows.append({
                'leading_item_id': pair_key[0],
                'following_item_id': pair_key[1],
                'value': avg_pred
            })
        model_submission = pd.DataFrame(model_submission_rows)
        
        if len(answer_df) > 0 and len(model_submission) > 0:
            score = comovement_score(answer_df, model_submission)
            f1 = comovement_f1(answer_df, model_submission)
            nmae = comovement_nmae(answer_df, model_submission)
            
            model_scores[model_name] = {
                'score': score,
                'f1': f1,
                'nmae': nmae
            }
            
            print(f"\n{model_name}:")
            print(f"  Score: {score:.6f}")
            print(f"  F1: {f1:.6f}")
            print(f"  NMAE: {nmae:.6f}")
    
    # 앙상블 성능 평가 (다양한 가중치 조합)
    print("\n[3단계] 앙상블 성능 평가 (다양한 가중치 조합)...")
    
    all_pairs = set()
    for model_name in model_predictions:
        all_pairs.update(model_predictions[model_name].keys())
    
    # 가중치 조합 테스트
    best_ensemble_score = -1
    best_ensemble_weights = None
    
    # 성능 기반 가중치 조합 테스트
    weight_combinations = [
        {'XGBoost': 0.35, 'LightGBM': 0.25, 'CatBoost': 0.40},  # train.py 현재 가중치
        {'XGBoost': 0.33, 'LightGBM': 0.33, 'CatBoost': 0.34},  # 균등
        {'XGBoost': 0.25, 'LightGBM': 0.40, 'CatBoost': 0.35},  # 이전 가중치
        {'XGBoost': 0.30, 'LightGBM': 0.35, 'CatBoost': 0.35},  # LightGBM 강조
        {'XGBoost': 0.40, 'LightGBM': 0.30, 'CatBoost': 0.30},  # XGBoost 강조
        {'XGBoost': 0.30, 'LightGBM': 0.30, 'CatBoost': 0.40},  # CatBoost 강조
    ]
    
    # 각 모델의 성능을 기반으로 가중치 계산
    if len(model_scores) == 3:
        scores = [model_scores['XGBoost']['score'], 
                  model_scores['LightGBM']['score'], 
                  model_scores['CatBoost']['score']]
        total_score = sum(scores)
        if total_score > 0:
            weight_combinations.append({
                'XGBoost': scores[0] / total_score,
                'LightGBM': scores[1] / total_score,
                'CatBoost': scores[2] / total_score
            })
    
    for weights in weight_combinations:
        ensemble_rows = []
        for pair_key in all_pairs:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
                if model_name in model_predictions and pair_key in model_predictions[model_name]:
                    model_preds = model_predictions[model_name][pair_key]
                    model_avg = np.mean(model_preds)
                    weight = weights[model_name]
                    weighted_sum += model_avg * weight
                    total_weight += weight
            
            avg_pred = weighted_sum / total_weight if total_weight > 0 else 0.0
            avg_pred = max(0.0, float(avg_pred))
            avg_pred = int(round(avg_pred))
            
            ensemble_rows.append({
                'leading_item_id': pair_key[0],
                'following_item_id': pair_key[1],
                'value': avg_pred
            })
        
        ensemble_submission = pd.DataFrame(ensemble_rows)
        
        if len(answer_df) > 0 and len(ensemble_submission) > 0:
            score = comovement_score(answer_df, ensemble_submission)
            f1 = comovement_f1(answer_df, ensemble_submission)
            nmae = comovement_nmae(answer_df, ensemble_submission)
            
            print(f"\n가중치 조합: XGB={weights['XGBoost']:.2f}, LGB={weights['LightGBM']:.2f}, CB={weights['CatBoost']:.2f}")
            print(f"  Score: {score:.6f}, F1: {f1:.6f}, NMAE: {nmae:.6f}")
            
            if score > best_ensemble_score:
                best_ensemble_score = score
                best_ensemble_weights = weights.copy()
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"\n최적 앙상블 가중치:")
    print(f"  XGBoost: {best_ensemble_weights['XGBoost']:.2f}")
    print(f"  LightGBM: {best_ensemble_weights['LightGBM']:.2f}")
    print(f"  CatBoost: {best_ensemble_weights['CatBoost']:.2f}")
    print(f"  최고 Score: {best_ensemble_score:.6f}")
    
    return best_ensemble_weights, model_scores


if __name__ == "__main__":
    # 데이터 로드
    train = pd.read_csv('../data/train.csv')
    monthly = (
        train
        .groupby(["item_id", "year", "month"], as_index=False)["value"]
        .sum()
    )
    monthly["ym"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )
    pivot = (
        monthly
        .pivot(index="item_id", columns="ym", values="value")
        .fillna(0.0)
    )
    
    # 검증 데이터 분할
    months = pivot.columns.to_list()
    n_months = len(months)
    split_idx = int(n_months * 0.9)
    train_months = months[:split_idx]
    val_months = months[split_idx:]
    
    pivot_train = pivot[train_months].copy()
    pivot_for_answer = pivot[months[:split_idx + len(val_months)]].copy()
    
    feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                    'b_trend', 'a_trend', 'b_ma3', 'b_change']
    
    # train.py와 동일한 설정 사용
    best_weights, model_scores = optimize_and_ensemble(
        pivot_train, pivot_for_answer, feature_cols,
        n_splits=5,  # train.py와 동일
        seeds=[42, 1031, 106]  # train.py와 동일한 seed 사용
    )

