"""
하이퍼파라미터 최적화 스크립트
- 상관계수 임계값 최적화
- min_nonzero 최적화
- lag 범위 최적화
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import warnings
from itertools import product

warnings.filterwarnings('ignore')

# train.py의 함수들을 재사용
sys.path.append(os.path.dirname(__file__))
from train import find_comovement_pairs

# evaluation.py import
sys.path.append(os.path.join(os.path.dirname(__file__), '../document'))
from evaluation import comovement_f1


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
    
    return pivot_train, pivot_for_answer


def optimize_correlation_threshold(pivot_train, pivot_for_answer):
    """상관계수 임계값 최적화"""
    print("\n" + "="*60)
    print("1. 상관계수 임계값 최적화")
    print("="*60)
    
    answer_pairs = find_comovement_pairs(pivot_for_answer, max_lag=6, min_nonzero=12, corr_threshold=0.35)
    answer_pairs_set = set(zip(answer_pairs['leading_item_id'], answer_pairs['following_item_id']))
    
    thresholds = np.arange(0.30, 0.50, 0.01)
    results = []
    
    for threshold in thresholds:
        pred_pairs = find_comovement_pairs(pivot_train, max_lag=6, min_nonzero=12, corr_threshold=threshold)
        pred_pairs_set = set(zip(pred_pairs['leading_item_id'], pred_pairs['following_item_id']))
        
        tp = len(pred_pairs_set & answer_pairs_set)
        fp = len(pred_pairs_set - answer_pairs_set)
        fn = len(answer_pairs_set - pred_pairs_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results.append({
            'threshold': threshold,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_pairs': len(pred_pairs)
        })
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_f1 = results_df.loc[best_idx, 'f1']
    
    print(f"\n최적 임계값: {best_threshold:.3f} (F1={best_f1:.4f})")
    print(f"  Precision: {results_df.loc[best_idx, 'precision']:.4f}")
    print(f"  Recall: {results_df.loc[best_idx, 'recall']:.4f}")
    print(f"  TP: {results_df.loc[best_idx, 'tp']}, FP: {results_df.loc[best_idx, 'fp']}, FN: {results_df.loc[best_idx, 'fn']}")
    
    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['threshold'], results_df['f1'], marker='o', label='F1 Score', linewidth=2)
    plt.plot(results_df['threshold'], results_df['precision'], marker='s', label='Precision', linewidth=2)
    plt.plot(results_df['threshold'], results_df['recall'], marker='^', label='Recall', linewidth=2)
    plt.axvline(best_threshold, color='r', linestyle='--', alpha=0.5, label=f'최적 ({best_threshold:.3f})')
    plt.xlabel('상관계수 임계값')
    plt.ylabel('점수')
    plt.title('상관계수 임계값 최적화')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../analysis/optimize_correlation_threshold.png', dpi=150, bbox_inches='tight')
    
    results_df.to_csv('../analysis/correlation_threshold_optimization.csv', index=False)
    
    return best_threshold, results_df


def optimize_min_nonzero(pivot_train, pivot_for_answer, best_corr_threshold):
    """min_nonzero 최적화"""
    print("\n" + "="*60)
    print("2. min_nonzero 최적화")
    print("="*60)
    
    answer_pairs = find_comovement_pairs(pivot_for_answer, max_lag=6, min_nonzero=12, corr_threshold=best_corr_threshold)
    answer_pairs_set = set(zip(answer_pairs['leading_item_id'], answer_pairs['following_item_id']))
    
    min_nonzero_values = [8, 10, 12, 14, 16, 18, 20]
    results = []
    
    for min_nonzero in min_nonzero_values:
        pred_pairs = find_comovement_pairs(pivot_train, max_lag=6, min_nonzero=min_nonzero, corr_threshold=best_corr_threshold)
        pred_pairs_set = set(zip(pred_pairs['leading_item_id'], pred_pairs['following_item_id']))
        
        tp = len(pred_pairs_set & answer_pairs_set)
        fp = len(pred_pairs_set - answer_pairs_set)
        fn = len(answer_pairs_set - pred_pairs_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results.append({
            'min_nonzero': min_nonzero,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_pairs': len(pred_pairs)
        })
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['f1'].idxmax()
    best_min_nonzero = results_df.loc[best_idx, 'min_nonzero']
    best_f1 = results_df.loc[best_idx, 'f1']
    
    print(f"\n최적 min_nonzero: {best_min_nonzero} (F1={best_f1:.4f})")
    print(f"  Precision: {results_df.loc[best_idx, 'precision']:.4f}")
    print(f"  Recall: {results_df.loc[best_idx, 'recall']:.4f}")
    
    results_df.to_csv('../analysis/min_nonzero_optimization.csv', index=False)
    
    return best_min_nonzero, results_df


def optimize_lag_range(pivot_train, pivot_for_answer, best_corr_threshold, best_min_nonzero):
    """lag 범위 최적화"""
    print("\n" + "="*60)
    print("3. Lag 범위 최적화")
    print("="*60)
    
    answer_pairs = find_comovement_pairs(pivot_for_answer, max_lag=6, min_nonzero=best_min_nonzero, corr_threshold=best_corr_threshold)
    answer_pairs_set = set(zip(answer_pairs['leading_item_id'], answer_pairs['following_item_id']))
    
    max_lag_values = [4, 5, 6, 7, 8]
    results = []
    
    for max_lag in max_lag_values:
        pred_pairs = find_comovement_pairs(pivot_train, max_lag=max_lag, min_nonzero=best_min_nonzero, corr_threshold=best_corr_threshold)
        pred_pairs_set = set(zip(pred_pairs['leading_item_id'], pred_pairs['following_item_id']))
        
        tp = len(pred_pairs_set & answer_pairs_set)
        fp = len(pred_pairs_set - answer_pairs_set)
        fn = len(answer_pairs_set - pred_pairs_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results.append({
            'max_lag': max_lag,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_pairs': len(pred_pairs)
        })
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['f1'].idxmax()
    best_max_lag = results_df.loc[best_idx, 'max_lag']
    best_f1 = results_df.loc[best_idx, 'f1']
    
    print(f"\n최적 max_lag: {best_max_lag} (F1={best_f1:.4f})")
    print(f"  Precision: {results_df.loc[best_idx, 'precision']:.4f}")
    print(f"  Recall: {results_df.loc[best_idx, 'recall']:.4f}")
    
    results_df.to_csv('../analysis/lag_range_optimization.csv', index=False)
    
    return best_max_lag, results_df


def main():
    """메인 실행 함수"""
    # 분석 결과 저장 디렉토리 생성
    os.makedirs('../analysis', exist_ok=True)
    
    print("="*60)
    print("하이퍼파라미터 최적화")
    print("="*60)
    
    # 데이터 로드 및 준비
    pivot = load_and_prepare_data()
    pivot_train, pivot_for_answer = create_validation_split(pivot)
    
    # 1. 상관계수 임계값 최적화
    best_corr_threshold, corr_results = optimize_correlation_threshold(pivot_train, pivot_for_answer)
    
    # 2. min_nonzero 최적화
    best_min_nonzero, min_nonzero_results = optimize_min_nonzero(pivot_train, pivot_for_answer, best_corr_threshold)
    
    # 3. lag 범위 최적화
    best_max_lag, lag_results = optimize_lag_range(pivot_train, pivot_for_answer, best_corr_threshold, best_min_nonzero)
    
    print("\n" + "="*60)
    print("최적화 완료!")
    print("="*60)
    print(f"\n최적 하이퍼파라미터:")
    print(f"  상관계수 임계값: {best_corr_threshold:.3f}")
    print(f"  min_nonzero: {best_min_nonzero}")
    print(f"  max_lag: {best_max_lag}")


if __name__ == "__main__":
    main()

