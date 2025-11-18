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


def optimize_and_ensemble(pivot_train, pivot_for_answer, feature_cols, n_splits=3):
    """
    각 모델을 독립적으로 최적화한 후 앙상블
    """
    print("=" * 60)
    print("단일 모델 최적화 후 앙상블")
    print("=" * 60)
    
    # 공행성 쌍 탐색
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
    
    # 각 모델별 최적 파라미터 (간단한 Grid Search 결과 또는 현재 파라미터)
    # 실제로는 optimize_hyperparameters_grid.py를 먼저 실행하여 최적 파라미터를 찾아야 함
    # 여기서는 현재 파라미터를 사용하고, 각 모델을 독립적으로 학습
    
    print("\n[1단계] 각 모델을 독립적으로 학습 및 예측...")
    
    models_config = [
        {
            'name': 'XGBoost',
            'type': 'xgb',
            'params': {
                'n_estimators': 250,
                'max_depth': 6,
                'learning_rate': 0.08,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'min_child_weight': 3,
                'reg_alpha': 0.05,
                'reg_lambda': 0.5,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
        },
        {
            'name': 'LightGBM',
            'type': 'lgb',
            'params': {
                'n_estimators': 250,
                'max_depth': 6,
                'learning_rate': 0.08,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'min_child_samples': 3,
                'reg_alpha': 0.05,
                'reg_lambda': 0.5,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1
            }
        },
        {
            'name': 'CatBoost',
            'type': 'cb',
            'params': {
                'iterations': 250,
                'depth': 6,
                'learning_rate': 0.08,
                'subsample': 0.85,
                'colsample_bylevel': 0.85,
                'min_data_in_leaf': 3,
                'l2_leaf_reg': 0.5,
                'random_state': 42,
                'thread_count': -1,
                'verbose': False
            }
        }
    ]
    
    # KFold로 각 모델 학습 및 예측
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_predictions = {}  # {model_name: {pair_key: [preds]}}
    
    for model_config in models_config:
        model_name = model_config['name']
        model_type = model_config['type']
        params = model_config['params']
        
        print(f"\n--- {model_name} 학습 중... ---")
        model_predictions[model_name] = {}
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            print(f"  Fold {fold_idx}/{n_splits}")
            
            X_train_fold = X[train_idx]
            y_train_fold = y_log[train_idx]
            
            # 모델 생성 및 학습
            if model_type == 'xgb':
                reg = xgb.XGBRegressor(**params)
            elif model_type == 'lgb':
                reg = lgb.LGBMRegressor(**params)
            elif model_type == 'cb':
                reg = cb.CatBoostRegressor(**params)
            
            reg.fit(X_train_fold, y_train_fold)
            
            # 예측
            fold_submission = predict(pivot_train, pairs, reg, feature_cols)
            
            # 예측 결과 누적
            for _, pred_row in fold_submission.iterrows():
                pair_key = (pred_row['leading_item_id'], pred_row['following_item_id'])
                if pair_key not in model_predictions[model_name]:
                    model_predictions[model_name][pair_key] = []
                model_predictions[model_name][pair_key].append(pred_row['value'])
    
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
        {'XGBoost': 0.25, 'LightGBM': 0.40, 'CatBoost': 0.35},  # 현재
        {'XGBoost': 0.33, 'LightGBM': 0.33, 'CatBoost': 0.34},  # 균등
        # 성능이 좋은 모델에 더 높은 가중치
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
    
    best_weights, model_scores = optimize_and_ensemble(
        pivot_train, pivot_for_answer, feature_cols
    )

