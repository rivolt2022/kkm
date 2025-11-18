"""
새로 추가된 피처들의 중요도와 상관관계 분석
성능 하락 원인 파악을 위한 분석 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# evaluation.py import를 위한 경로 추가
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../document'))
from evaluation import comovement_score, comovement_f1, comovement_nmae

from train import (
    find_comovement_pairs, build_training_data,
    compute_item_statistics, cross_validate_with_kfold
)


def analyze_feature_correlation():
    """피처 간 상관관계 분석"""
    print("=" * 60)
    print("[1단계] 피처 간 상관관계 분석")
    print("=" * 60)
    
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
    
    # 공행성 쌍 탐색
    pairs = find_comovement_pairs(pivot, max_lag=6, min_nonzero=12, corr_threshold=0.34)
    
    # 학습 데이터 생성
    item_stats = compute_item_statistics(pivot)
    df_train = build_training_data(pivot, pairs, item_stats=item_stats)
    
    # 피처 컬럼만 선택
    feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                    'b_trend', 'a_trend', 'b_ma3', 'b_change',
                    'lag_x_corr', 'b_ma6', 'b_std3',
                    'leader_mean', 'leader_std', 'follower_mean', 'follower_std']
    
    df_features = df_train[feature_cols]
    
    # 상관관계 행렬 계산
    corr_matrix = df_features.corr()
    
    print("\n피처 간 상관관계 행렬:")
    print(corr_matrix.round(3))
    
    # 상관관계 히트맵 시각화
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('피처 간 상관관계 히트맵', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('../analysis/feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n상관관계 히트맵 저장: ../analysis/feature_correlation_heatmap.png")
    
    # 높은 상관관계를 가진 피처 쌍 찾기
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:  # 0.8 이상의 상관관계
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    if len(high_corr_pairs) > 0:
        print("\n높은 상관관계를 가진 피처 쌍 (|corr| > 0.8):")
        high_corr_df = pd.DataFrame(high_corr_pairs)
        print(high_corr_df.to_string(index=False))
        high_corr_df.to_csv('../analysis/high_correlation_pairs.csv', index=False, encoding='utf-8-sig')
    
    return corr_matrix, df_features


def analyze_feature_importance():
    """피처 중요도 분석 (XGBoost 사용)"""
    print("\n" + "=" * 60)
    print("[2단계] 피처 중요도 분석")
    print("=" * 60)
    
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    
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
    
    # 공행성 쌍 탐색
    pairs = find_comovement_pairs(pivot, max_lag=6, min_nonzero=12, corr_threshold=0.34)
    
    # 학습 데이터 생성
    item_stats = compute_item_statistics(pivot)
    df_train = build_training_data(pivot, pairs, item_stats=item_stats)
    
    # 피처 컬럼
    feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                    'b_trend', 'a_trend', 'b_ma3', 'b_change',
                    'lag_x_corr', 'b_ma6', 'b_std3',
                    'leader_mean', 'leader_std', 'follower_mean', 'follower_std']
    
    X = df_train[feature_cols].values
    y = df_train["target"].values
    y_log = np.log1p(y)
    
    # 학습/검증 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    # XGBoost 모델 학습
    reg = xgb.XGBRegressor(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    reg.fit(X_train, y_train)
    
    # 피처 중요도 추출
    importance = reg.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\n피처 중요도 (내림차순):")
    print(feature_importance_df.to_string(index=False))
    
    # 피처 중요도 시각화
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'], edgecolor='black')
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
    plt.xlabel('중요도')
    plt.title('피처 중요도 (XGBoost)', fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../analysis/feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n피처 중요도 그래프 저장: ../analysis/feature_importance.png")
    
    # CSV로 저장
    feature_importance_df.to_csv('../analysis/feature_importance.csv', index=False, encoding='utf-8-sig')
    
    return feature_importance_df


def compare_feature_sets():
    """기존 피처 세트 vs 새로운 피처 세트 성능 비교"""
    print("\n" + "=" * 60)
    print("[3단계] 피처 세트별 성능 비교")
    print("=" * 60)
    
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
    
    months = pivot.columns.to_list()
    n_months = len(months)
    
    # 검증 모드: 데이터 분할
    split_idx = int(n_months * 0.9)
    train_months = months[:split_idx]
    val_months = months[split_idx:]
    
    pivot_train = pivot[train_months].copy()
    pivot_for_answer = pivot[months[:split_idx + len(val_months)]].copy()
    predict_month_idx = split_idx
    
    # 공행성 쌍 탐색
    corr_threshold = 0.34
    min_nonzero = 12
    pairs = find_comovement_pairs(pivot_train, max_lag=6, min_nonzero=min_nonzero, corr_threshold=corr_threshold)
    answer_pairs = find_comovement_pairs(pivot_for_answer, max_lag=6, min_nonzero=min_nonzero, corr_threshold=corr_threshold)
    
    # 정답 데이터 생성
    answer_rows = []
    for _, row in answer_pairs.iterrows():
        following_item_id = row['following_item_id']
        if following_item_id in pivot.index and predict_month_idx < n_months:
            actual_value = pivot.loc[following_item_id, months[predict_month_idx]]
            answer_rows.append({
                'leading_item_id': row['leading_item_id'],
                'following_item_id': following_item_id,
                'value': int(round(float(actual_value)))
            })
    answer_df = pd.DataFrame(answer_rows)
    
    # 피처 세트 정의
    feature_sets = {
        '기존 피처 (9개)': ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                          'b_trend', 'a_trend', 'b_ma3', 'b_change'],
        '기존 + 상호작용 (10개)': ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                                 'b_trend', 'a_trend', 'b_ma3', 'b_change', 'lag_x_corr'],
        '기존 + 시계열 확장 (12개)': ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                                    'b_trend', 'a_trend', 'b_ma3', 'b_change', 
                                    'b_ma6', 'b_std3'],
        '기존 + 통계량 (13개)': ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                               'b_trend', 'a_trend', 'b_ma3', 'b_change',
                               'leader_mean', 'leader_std', 'follower_mean', 'follower_std'],
        '전체 피처 (16개)': ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                          'b_trend', 'a_trend', 'b_ma3', 'b_change',
                          'lag_x_corr', 'b_ma6', 'b_std3',
                          'leader_mean', 'leader_std', 'follower_mean', 'follower_std'],
    }
    
    results = []
    
    for set_name, feature_cols in feature_sets.items():
        print(f"\n--- {set_name} 테스트 중... ---")
        
        # KFold 교차 검증 (간단하게 3-Fold로 빠르게 테스트)
        try:
            submission, fold_scores = cross_validate_with_kfold(
                pivot_train, pairs, feature_cols,
                is_validate_mode=True, answer_df=answer_df, n_splits=3
            )
            
            if len(answer_df) > 0 and len(submission) > 0:
                score = comovement_score(answer_df, submission)
                f1 = comovement_f1(answer_df, submission)
                nmae = comovement_nmae(answer_df, submission)
                
                results.append({
                    'feature_set': set_name,
                    'num_features': len(feature_cols),
                    'f1': f1,
                    'nmae': nmae,
                    'score': score
                })
                
                print(f"  F1: {f1:.6f}, NMAE: {nmae:.6f}, Score: {score:.6f}")
        except Exception as e:
            print(f"  오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        print("\n" + "=" * 60)
        print("피처 세트별 성능 비교 결과")
        print("=" * 60)
        print(results_df.to_string(index=False))
        results_df.to_csv('../analysis/feature_set_comparison.csv', index=False, encoding='utf-8-sig')
        
        # 성능 비교 시각화
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].bar(range(len(results_df)), results_df['f1'], edgecolor='black')
        axes[0].set_xticks(range(len(results_df)))
        axes[0].set_xticklabels(results_df['feature_set'], rotation=45, ha='right')
        axes[0].set_ylabel('F1 Score')
        axes[0].set_title('F1 Score 비교')
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(range(len(results_df)), results_df['nmae'], edgecolor='black')
        axes[1].set_xticks(range(len(results_df)))
        axes[1].set_xticklabels(results_df['feature_set'], rotation=45, ha='right')
        axes[1].set_ylabel('NMAE')
        axes[1].set_title('NMAE 비교 (낮을수록 좋음)')
        axes[1].grid(axis='y', alpha=0.3)
        
        axes[2].bar(range(len(results_df)), results_df['score'], edgecolor='black')
        axes[2].set_xticks(range(len(results_df)))
        axes[2].set_xticklabels(results_df['feature_set'], rotation=45, ha='right')
        axes[2].set_ylabel('Final Score')
        axes[2].set_title('Final Score 비교')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../analysis/feature_set_comparison.png', dpi=300, bbox_inches='tight')
        print("\n피처 세트 비교 그래프 저장: ../analysis/feature_set_comparison.png")
    
    return results_df


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("새로 추가된 피처 분석 - 성능 하락 원인 파악")
    print("=" * 60)
    
    # 1. 피처 간 상관관계 분석
    corr_matrix, df_features = analyze_feature_correlation()
    
    # 2. 피처 중요도 분석
    feature_importance_df = analyze_feature_importance()
    
    # 3. 피처 세트별 성능 비교
    results_df = compare_feature_sets()
    
    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)
    print("\n생성된 파일:")
    print("  - ../analysis/feature_correlation_heatmap.png")
    print("  - ../analysis/high_correlation_pairs.csv")
    print("  - ../analysis/feature_importance.png")
    print("  - ../analysis/feature_importance.csv")
    print("  - ../analysis/feature_set_comparison.png")
    print("  - ../analysis/feature_set_comparison.csv")


if __name__ == "__main__":
    main()

