"""
작은 범위의 Grid Search를 이용한 하이퍼파라미터 최적화
Optuna 대신 작은 범위에서 Grid Search를 사용하여 안정적인 최적화
"""

import pandas as pd
import numpy as np
import sys
import os
from itertools import product
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
from sklearn.model_selection import train_test_split


def optimize_single_model_hyperparameters(pivot_train, pivot_for_answer, feature_cols, 
                                         model_type='xgb', n_splits=3):
    """
    단일 모델의 하이퍼파라미터를 Grid Search로 최적화
    작은 범위에서만 탐색하여 안정성 확보
    """
    print("=" * 60)
    print(f"{model_type.upper()} 하이퍼파라미터 최적화 (Grid Search)")
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
    
    # 현재 하이퍼파라미터 (기준점)
    if model_type == 'xgb':
        base_params = {
            'n_estimators': 250,
            'max_depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 3,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
        }
        # 작은 범위로 탐색 (±10%)
        param_grid = {
            'max_depth': [5, 6, 7],
            'learning_rate': [0.07, 0.08, 0.09],
            'subsample': [0.80, 0.85, 0.90],
            'colsample_bytree': [0.80, 0.85, 0.90],
            'reg_alpha': [0.03, 0.05, 0.07],
            'reg_lambda': [0.4, 0.5, 0.6],
        }
    elif model_type == 'lgb':
        base_params = {
            'n_estimators': 250,
            'max_depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_samples': 3,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
        }
        param_grid = {
            'max_depth': [5, 6, 7],
            'learning_rate': [0.07, 0.08, 0.09],
            'subsample': [0.80, 0.85, 0.90],
            'colsample_bytree': [0.80, 0.85, 0.90],
            'reg_alpha': [0.03, 0.05, 0.07],
            'reg_lambda': [0.4, 0.5, 0.6],
        }
    elif model_type == 'cb':
        base_params = {
            'iterations': 250,
            'depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.85,
            'colsample_bylevel': 0.85,
            'min_data_in_leaf': 3,
            'l2_leaf_reg': 0.5,
        }
        param_grid = {
            'depth': [5, 6, 7],
            'learning_rate': [0.07, 0.08, 0.09],
            'subsample': [0.80, 0.85, 0.90],
            'colsample_bylevel': [0.80, 0.85, 0.90],
            'l2_leaf_reg': [0.4, 0.5, 0.6],
        }
    
    # Grid Search 수행
    print(f"\n총 {np.prod([len(v) for v in param_grid.values()])}개의 조합을 테스트합니다.")
    
    best_score = -1
    best_params = None
    results = []
    
    # 모든 조합 생성
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for param_combo in tqdm(product(*param_values), total=np.prod([len(v) for v in param_values])):
        params = dict(zip(param_names, param_combo))
        
        # base_params와 병합
        full_params = base_params.copy()
        full_params.update(params)
        
        # 간단한 train/val split으로 빠르게 테스트
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_log, test_size=0.2, random_state=42
        )
        
        try:
            # 모델 학습
            if model_type == 'xgb':
                reg = xgb.XGBRegressor(
                    **full_params,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
            elif model_type == 'lgb':
                reg = lgb.LGBMRegressor(
                    **full_params,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )
            elif model_type == 'cb':
                reg = cb.CatBoostRegressor(
                    **full_params,
                    random_state=42,
                    thread_count=-1,
                    verbose=False
                )
            
            reg.fit(X_train, y_train)
            
            # 예측
            submission = predict(pivot_train, pairs, reg, feature_cols)
            
            if len(answer_df) > 0 and len(submission) > 0:
                score = comovement_score(answer_df, submission)
                f1 = comovement_f1(answer_df, submission)
                nmae = comovement_nmae(answer_df, submission)
                
                results.append({
                    **params,
                    'score': score,
                    'f1': f1,
                    'nmae': nmae
                })
                
                if score > best_score:
                    best_score = score
                    best_params = full_params.copy()
        except Exception as e:
            print(f"  오류 발생: {e}")
            continue
    
    # 결과 정리
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        print("\n" + "=" * 60)
        print(f"{model_type.upper()} 하이퍼파라미터 최적화 결과")
        print("=" * 60)
        print(f"\n최적 파라미터:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\n최고 Score: {best_score:.6f}")
        
        print(f"\n상위 10개 파라미터 조합:")
        print(results_df.head(10).to_string(index=False))
        
        # CSV로 저장
        results_df.to_csv(f'../analysis/{model_type}_hyperparameters_optimization.csv', 
                         index=False, encoding='utf-8-sig')
        print(f"\n결과 저장: ../analysis/{model_type}_hyperparameters_optimization.csv")
        
        return best_params, results_df
    else:
        print("\n경고: 유효한 결과가 없습니다.")
        return None, None


def main():
    """메인 실행 함수"""
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
    
    # 각 모델별로 최적화
    all_best_params = {}
    
    for model_type in ['xgb', 'lgb', 'cb']:
        best_params, results_df = optimize_single_model_hyperparameters(
            pivot_train, pivot_for_answer, feature_cols, model_type=model_type
        )
        if best_params:
            all_best_params[model_type] = best_params
    
    # 최종 결과 출력
    if all_best_params:
        print("\n" + "=" * 60)
        print("전체 모델 최적화 완료!")
        print("=" * 60)
        print("\n최적 파라미터를 train.py에 적용하려면:")
        for model_type, params in all_best_params.items():
            print(f"\n{model_type.upper()}:")
            for key, value in params.items():
                print(f"  {key}: {value},")


if __name__ == "__main__":
    main()

