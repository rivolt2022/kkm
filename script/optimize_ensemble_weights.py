"""
앙상블 가중치 최적화 스크립트
Optuna 대신 앙상블 가중치를 최적화하는 것이 더 효과적이고 안정적
"""

import pandas as pd
import numpy as np
import sys
import os
from itertools import product

# evaluation.py import를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../document'))
from evaluation import comovement_score, comovement_f1, comovement_nmae

from train import (
    find_comovement_pairs, cross_validate_with_kfold
)


def optimize_ensemble_weights(pivot_train, pivot_for_answer, feature_cols):
    """
    앙상블 가중치 최적화
    모델별 가중치를 조정하여 최종 Score를 최대화
    """
    print("=" * 60)
    print("앙상블 가중치 최적화")
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
    
    print(f"\n정답 데이터 수: {len(answer_df)}")
    
    # 가중치 조합 탐색 (Grid Search)
    print("\n[가중치 조합 탐색 중...]")
    print("각 가중치를 0.05 단위로 탐색 (합이 1.0이 되도록)")
    
    best_score = -1
    best_weights = None
    results = []
    
    # 가중치 범위: 각 모델별로 0.15 ~ 0.50 (합이 1.0이 되도록)
    # XGBoost, LightGBM, CatBoost 가중치
    weights_list = []
    for xgb_w in np.arange(0.15, 0.51, 0.05):
        for lgb_w in np.arange(0.15, 0.51, 0.05):
            cb_w = 1.0 - xgb_w - lgb_w
            if 0.15 <= cb_w <= 0.50:  # CatBoost 가중치도 범위 내
                weights_list.append({
                    'XGBoost': round(xgb_w, 2),
                    'LightGBM': round(lgb_w, 2),
                    'CatBoost': round(cb_w, 2)
                })
    
    print(f"총 {len(weights_list)}개의 가중치 조합을 테스트합니다.")
    
    for idx, weights in enumerate(weights_list, 1):
        print(f"\n[{idx}/{len(weights_list)}] 테스트 중: XGB={weights['XGBoost']:.2f}, LGB={weights['LightGBM']:.2f}, CB={weights['CatBoost']:.2f}")
        
        try:
            # KFold 교차 검증 수행 (가중치 적용)
            submission, fold_scores = cross_validate_with_kfold(
                pivot_train, pairs, feature_cols,
                is_validate_mode=True, answer_df=answer_df, n_splits=3,  # 빠른 테스트를 위해 3-Fold
                model_weights=weights
            )
            
            if len(answer_df) > 0 and len(submission) > 0:
                score = comovement_score(answer_df, submission)
                f1 = comovement_f1(answer_df, submission)
                nmae = comovement_nmae(answer_df, submission)
                
                results.append({
                    'xgb_weight': weights['XGBoost'],
                    'lgb_weight': weights['LightGBM'],
                    'cb_weight': weights['CatBoost'],
                    'score': score,
                    'f1': f1,
                    'nmae': nmae
                })
                
                print(f"  결과: Score={score:.6f}, F1={f1:.6f}, NMAE={nmae:.6f}")
                
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                    print(f"  ★ 새로운 최고 점수! ★")
        except Exception as e:
            print(f"  오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # 결과 정리
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        print("\n" + "=" * 60)
        print("앙상블 가중치 최적화 결과")
        print("=" * 60)
        print(f"\n최적 가중치:")
        print(f"  XGBoost: {best_weights['XGBoost']:.2f}")
        print(f"  LightGBM: {best_weights['LightGBM']:.2f}")
        print(f"  CatBoost: {best_weights['CatBoost']:.2f}")
        print(f"  최고 Score: {best_score:.6f}")
        
        print(f"\n상위 10개 가중치 조합:")
        print(results_df.head(10).to_string(index=False))
        
        # CSV로 저장
        results_df.to_csv('../analysis/ensemble_weights_optimization.csv', index=False, encoding='utf-8-sig')
        print(f"\n결과 저장: ../analysis/ensemble_weights_optimization.csv")
        
        return best_weights, results_df
    else:
        print("\n경고: 유효한 결과가 없습니다.")
        return None, None


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
    
    best_weights, results_df = optimize_ensemble_weights(pivot_train, pivot_for_answer, feature_cols)
    
    if best_weights:
        print("\n" + "=" * 60)
        print("최적 가중치를 train.py에 적용하려면:")
        print("=" * 60)
        print(f"model_weights = {{'LightGBM': {best_weights['LightGBM']:.2f}, 'CatBoost': {best_weights['CatBoost']:.2f}, 'XGBoost': {best_weights['XGBoost']:.2f}}}")
