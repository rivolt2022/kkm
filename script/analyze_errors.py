"""
예측 오차 분석 스크립트
- 어떤 쌍/조건에서 오차가 큰지 분석
- 과소/과대 예측 패턴 분석
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# train.py의 함수들을 재사용
sys.path.append(os.path.dirname(__file__))
from train import find_comovement_pairs, build_training_data, predict
import xgboost as xgb

# evaluation.py import
sys.path.append(os.path.join(os.path.dirname(__file__), '../document'))
from evaluation import comovement_score, comovement_f1, comovement_nmae


def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    print("데이터 로드 중...")
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
    
    return pivot


def create_validation_split(pivot):
    """검증 모드와 동일하게 데이터 분할"""
    months = pivot.columns.to_list()
    n_months = len(months)
    split_idx = int(n_months * 0.9)
    
    train_months = months[:split_idx]
    val_months = months[split_idx:]
    
    pivot_train = pivot[train_months].copy()
    pivot_for_answer = pivot[months[:split_idx + len(val_months)]].copy()
    
    return pivot_train, pivot_for_answer, split_idx, months


def analyze_error_patterns(pivot_train, pivot_for_answer, split_idx, months):
    """오차 패턴 상세 분석"""
    print("\n" + "="*60)
    print("예측 오차 패턴 분석")
    print("="*60)
    
    # 공행성 쌍 탐색
    pairs = find_comovement_pairs(pivot_train, max_lag=6, min_nonzero=12, corr_threshold=0.35)
    answer_pairs = find_comovement_pairs(pivot_for_answer, max_lag=6, min_nonzero=12, corr_threshold=0.35)
    
    # 학습 데이터 생성
    feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                    'b_trend', 'a_trend', 'b_ma3', 'b_change']
    df_train = build_training_data(pivot_train, pairs)
    
    if len(df_train) == 0:
        print("경고: 학습 데이터가 없습니다.")
        return None
    
    train_X = df_train[feature_cols].values
    train_y = df_train["target"].values
    
    # 모델 학습
    reg = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    reg.fit(train_X, train_y)
    
    # 예측
    submission = predict(pivot_train, pairs, reg, feature_cols)
    
    # 정답 생성
    answer_rows = []
    for _, row in answer_pairs.iterrows():
        following_item_id = row['following_item_id']
        if following_item_id in pivot_for_answer.index and split_idx < len(months):
            actual_value = pivot_for_answer.loc[following_item_id, months[split_idx]]
            answer_rows.append({
                'leading_item_id': row['leading_item_id'],
                'following_item_id': following_item_id,
                'value': int(round(float(actual_value)))
            })
    
    answer_df = pd.DataFrame(answer_rows)
    
    # 쌍 정보와 오차 결합
    pairs_dict = dict(zip(
        zip(pairs['leading_item_id'], pairs['following_item_id']),
        zip(pairs['max_corr'], pairs['best_lag'])
    ))
    
    pred_dict = dict(zip(
        zip(submission['leading_item_id'], submission['following_item_id']),
        submission['value']
    ))
    answer_dict = dict(zip(
        zip(answer_df['leading_item_id'], answer_df['following_item_id']),
        answer_df['value']
    ))
    
    error_details = []
    
    for pair in set(list(pred_dict.keys()) + list(answer_dict.keys())):
        leader, follower = pair
        
        if pair in pred_dict and pair in answer_dict:
            y_true = answer_dict[pair]
            y_pred = pred_dict[pair]
            
            if y_true > 0:
                rel_error = abs(y_true - y_pred) / y_true
                rel_error = min(rel_error, 1.0)
            else:
                rel_error = 1.0 if y_pred > 0 else 0.0
            
            # 쌍 특성
            corr, lag = pairs_dict.get(pair, (0, 0))
            
            # 품목 특성
            if leader in pivot_train.index and follower in pivot_train.index:
                leader_series = pivot_train.loc[leader].values.astype(float)
                follower_series = pivot_train.loc[follower].values.astype(float)
                
                leader_mean = np.mean(leader_series[leader_series > 0]) if np.any(leader_series > 0) else 0
                follower_mean = np.mean(follower_series[follower_series > 0]) if np.any(follower_series > 0) else 0
                leader_cv = np.std(leader_series[leader_series > 0]) / leader_mean if leader_mean > 0 else 0
                follower_cv = np.std(follower_series[follower_series > 0]) / follower_mean if follower_mean > 0 else 0
            else:
                leader_mean = follower_mean = leader_cv = follower_cv = 0
            
            error_details.append({
                'leading_item_id': leader,
                'following_item_id': follower,
                'y_true': y_true,
                'y_pred': y_pred,
                'abs_error': abs(y_true - y_pred),
                'rel_error': rel_error,
                'is_overestimate': y_pred > y_true,
                'corr': corr,
                'lag': lag,
                'leader_mean': leader_mean,
                'follower_mean': follower_mean,
                'leader_cv': leader_cv,
                'follower_cv': follower_cv,
                'mean_ratio': follower_mean / leader_mean if leader_mean > 0 else 0,
            })
        elif pair in pred_dict:
            # FP 쌍
            error_details.append({
                'leading_item_id': leader,
                'following_item_id': follower,
                'y_true': 0,
                'y_pred': pred_dict[pair],
                'abs_error': pred_dict[pair],
                'rel_error': 1.0,
                'is_overestimate': True,
                'is_fp': True,
            })
        elif pair in answer_dict:
            # FN 쌍
            error_details.append({
                'leading_item_id': leader,
                'following_item_id': follower,
                'y_true': answer_dict[pair],
                'y_pred': 0,
                'abs_error': answer_dict[pair],
                'rel_error': 1.0,
                'is_overestimate': False,
                'is_fn': True,
            })
    
    error_df = pd.DataFrame(error_details)
    
    if len(error_df) > 0:
        # 오차가 큰 쌍 분석
        high_error = error_df[error_df['rel_error'] > 0.5].copy()
        
        print(f"\n전체 오차 통계:")
        print(f"평균 상대 오차: {error_df['rel_error'].mean():.4f}")
        print(f"중앙값 상대 오차: {error_df['rel_error'].median():.4f}")
        print(f"오차 > 50%인 쌍: {len(high_error)}개 ({len(high_error)/len(error_df)*100:.1f}%)")
        
        if 'corr' in high_error.columns:
            print(f"\n높은 오차 쌍의 특성:")
            print(f"평균 상관계수: {high_error['corr'].abs().mean():.4f}")
            print(f"평균 lag: {high_error['lag'].mean():.2f}")
            print(f"평균 변동계수 (follower): {high_error['follower_cv'].mean():.4f}")
        
        # 오차와 상관계수 관계
        if 'corr' in error_df.columns:
            plt.figure(figsize=(12, 6))
            plt.scatter(error_df['corr'].abs(), error_df['rel_error'], alpha=0.3, s=10)
            plt.xlabel('절댓값 상관계수')
            plt.ylabel('상대 오차')
            plt.title('상관계수와 예측 오차의 관계')
            plt.grid(True, alpha=0.3)
            plt.savefig('../analysis/correlation_vs_error.png', dpi=150, bbox_inches='tight')
            print("\n그래프 저장: analysis/correlation_vs_error.png")
        
        error_df.to_csv('../analysis/error_details.csv', index=False)
        print("상세 오차 데이터 저장: analysis/error_details.csv")
    
    return error_df


def main():
    """메인 실행 함수"""
    # 분석 결과 저장 디렉토리 생성
    os.makedirs('../analysis', exist_ok=True)
    
    print("="*60)
    print("예측 오차 분석")
    print("="*60)
    
    # 데이터 로드 및 준비
    pivot = load_and_prepare_data()
    pivot_train, pivot_for_answer, split_idx, months = create_validation_split(pivot)
    
    # 오차 패턴 분석
    error_df = analyze_error_patterns(pivot_train, pivot_for_answer, split_idx, months)
    
    print("\n" + "="*60)
    print("분석 완료!")
    print("="*60)


if __name__ == "__main__":
    main()

