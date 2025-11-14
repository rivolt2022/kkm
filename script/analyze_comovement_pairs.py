"""
공행성 쌍 탐색 개선 분석 스크립트
- 상관계수 분포 분석 (정답 쌍 vs FP 쌍)
- lag 분포 분석
- FP 쌍의 특성 분석
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행 가능하도록
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# train.py의 함수들을 재사용
sys.path.append(os.path.dirname(__file__))
from train import safe_corr, find_comovement_pairs

# evaluation.py import
sys.path.append(os.path.join(os.path.dirname(__file__), '../document'))
from evaluation import comovement_f1


def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    print("데이터 로드 중...")
    train = pd.read_csv('../data/train.csv')
    
    # 월별 피벗 테이블 생성
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


def analyze_correlation_distribution(pivot_train, pivot_for_answer, split_idx, months):
    """상관계수 분포 분석"""
    print("\n" + "="*60)
    print("1. 상관계수 분포 분석")
    print("="*60)
    
    # 학습 데이터로 예측할 공행성 쌍 탐색
    print("학습 데이터로 공행성 쌍 탐색 중...")
    pred_pairs = find_comovement_pairs(pivot_train, max_lag=6, min_nonzero=12, corr_threshold=0.35)
    print(f"예측 쌍 수: {len(pred_pairs)}")
    
    # 정답 공행성 쌍 탐색
    print("정답 공행성 쌍 탐색 중...")
    answer_pairs = find_comovement_pairs(pivot_for_answer, max_lag=6, min_nonzero=12, corr_threshold=0.35)
    print(f"정답 쌍 수: {len(answer_pairs)}")
    
    # 쌍 분류
    pred_pairs_set = set(zip(pred_pairs['leading_item_id'], pred_pairs['following_item_id']))
    answer_pairs_set = set(zip(answer_pairs['leading_item_id'], answer_pairs['following_item_id']))
    
    tp_pairs = pred_pairs_set & answer_pairs_set
    fp_pairs = pred_pairs_set - answer_pairs_set
    fn_pairs = answer_pairs_set - pred_pairs_set
    
    print(f"\nTP: {len(tp_pairs)}, FP: {len(fp_pairs)}, FN: {len(fn_pairs)}")
    
    # TP, FP, FN의 상관계수 분포 분석
    pred_dict = dict(zip(
        zip(pred_pairs['leading_item_id'], pred_pairs['following_item_id']),
        zip(pred_pairs['max_corr'], pred_pairs['best_lag'])
    ))
    
    tp_corrs = [pred_dict[pair][0] for pair in tp_pairs if pair in pred_dict]
    fp_corrs = [pred_dict[pair][0] for pair in fp_pairs if pair in pred_dict]
    fn_corrs = []
    for pair in fn_pairs:
        if pair in pred_dict:
            fn_corrs.append(pred_dict[pair][0])
    
    print(f"\n상관계수 통계:")
    print(f"TP 평균: {np.mean(tp_corrs):.4f}, 표준편차: {np.std(tp_corrs):.4f}")
    print(f"FP 평균: {np.mean(fp_corrs):.4f}, 표준편차: {np.std(fp_corrs):.4f}")
    if len(fn_corrs) > 0:
        print(f"FN 평균: {np.mean(fn_corrs):.4f}, 표준편차: {np.std(fn_corrs):.4f}")
    
    # 상관계수 분포 시각화
    plt.figure(figsize=(12, 6))
    plt.hist([abs(c) for c in tp_corrs], bins=50, alpha=0.7, label='TP', density=True)
    plt.hist([abs(c) for c in fp_corrs], bins=50, alpha=0.7, label='FP', density=True)
    if len(fn_corrs) > 0:
        plt.hist([abs(c) for c in fn_corrs], bins=50, alpha=0.7, label='FN', density=True)
    plt.xlabel('절댓값 상관계수')
    plt.ylabel('밀도')
    plt.title('상관계수 분포 비교 (TP vs FP vs FN)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../analysis/correlation_distribution.png', dpi=150, bbox_inches='tight')
    print("\n그래프 저장: analysis/correlation_distribution.png")
    
    return pred_pairs, answer_pairs, tp_pairs, fp_pairs, fn_pairs


def analyze_lag_distribution(pred_pairs, answer_pairs, tp_pairs, fp_pairs):
    """lag 분포 분석"""
    print("\n" + "="*60)
    print("2. Lag 분포 분석")
    print("="*60)
    
    pred_dict = dict(zip(
        zip(pred_pairs['leading_item_id'], pred_pairs['following_item_id']),
        zip(pred_pairs['max_corr'], pred_pairs['best_lag'])
    ))
    
    answer_dict = dict(zip(
        zip(answer_pairs['leading_item_id'], answer_pairs['following_item_id']),
        zip(answer_pairs['max_corr'], answer_pairs['best_lag'])
    ))
    
    tp_lags = [pred_dict[pair][1] for pair in tp_pairs if pair in pred_dict]
    fp_lags = [pred_dict[pair][1] for pair in fp_pairs if pair in pred_dict]
    
    print(f"\nLag 통계:")
    print(f"TP lag 평균: {np.mean(tp_lags):.2f}, 최빈값: {pd.Series(tp_lags).mode().values[0] if len(tp_lags) > 0 else 'N/A'}")
    print(f"FP lag 평균: {np.mean(fp_lags):.2f}, 최빈값: {pd.Series(fp_lags).mode().values[0] if len(fp_lags) > 0 else 'N/A'}")
    
    # Lag 분포 시각화
    plt.figure(figsize=(12, 6))
    tp_lag_counts = pd.Series(tp_lags).value_counts().sort_index()
    fp_lag_counts = pd.Series(fp_lags).value_counts().sort_index()
    
    x = range(1, 7)
    tp_counts = [tp_lag_counts.get(i, 0) for i in x]
    fp_counts = [fp_lag_counts.get(i, 0) for i in x]
    
    x_pos = np.arange(len(x))
    width = 0.35
    
    plt.bar(x_pos - width/2, tp_counts, width, label='TP', alpha=0.7)
    plt.bar(x_pos + width/2, fp_counts, width, label='FP', alpha=0.7)
    plt.xlabel('Lag')
    plt.ylabel('개수')
    plt.title('Lag 분포 비교 (TP vs FP)')
    plt.xticks(x_pos, x)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('../analysis/lag_distribution.png', dpi=150, bbox_inches='tight')
    print("\n그래프 저장: analysis/lag_distribution.png")
    
    return pred_dict, answer_dict


def analyze_fp_characteristics(pivot_train, fp_pairs, pred_dict):
    """FP 쌍의 특성 분석"""
    print("\n" + "="*60)
    print("3. FP 쌍 특성 분석")
    print("="*60)
    
    fp_analysis = []
    
    for pair in list(fp_pairs)[:100]:  # 처음 100개만 분석
        if pair not in pred_dict:
            continue
            
        leader, follower = pair
        corr, lag = pred_dict[pair]
        
        if leader not in pivot_train.index or follower not in pivot_train.index:
            continue
        
        leader_series = pivot_train.loc[leader].values.astype(float)
        follower_series = pivot_train.loc[follower].values.astype(float)
        
        # 통계 특성 계산
        leader_mean = np.mean(leader_series[leader_series > 0]) if np.any(leader_series > 0) else 0
        follower_mean = np.mean(follower_series[follower_series > 0]) if np.any(follower_series > 0) else 0
        leader_std = np.std(leader_series[leader_series > 0]) if np.any(leader_series > 0) else 0
        follower_std = np.std(follower_series[follower_series > 0]) if np.any(follower_series > 0) else 0
        leader_cv = leader_std / leader_mean if leader_mean > 0 else 0
        follower_cv = follower_std / follower_mean if follower_mean > 0 else 0
        
        fp_analysis.append({
            'leading_item_id': leader,
            'following_item_id': follower,
            'corr': corr,
            'lag': lag,
            'leader_mean': leader_mean,
            'follower_mean': follower_mean,
            'leader_cv': leader_cv,
            'follower_cv': follower_cv,
            'mean_ratio': follower_mean / leader_mean if leader_mean > 0 else 0,
        })
    
    fp_df = pd.DataFrame(fp_analysis)
    
    if len(fp_df) > 0:
        print(f"\nFP 쌍 특성 통계 (샘플 {len(fp_df)}개):")
        print(f"평균 상관계수: {fp_df['corr'].mean():.4f}")
        print(f"평균 lag: {fp_df['lag'].mean():.2f}")
        print(f"평균 변동계수 (leader): {fp_df['leader_cv'].mean():.4f}")
        print(f"평균 변동계수 (follower): {fp_df['follower_cv'].mean():.4f}")
        
        # 상관계수가 낮은 FP 쌍 비율
        low_corr_fp = (fp_df['corr'].abs() < 0.4).sum()
        print(f"\n상관계수 < 0.4인 FP 쌍: {low_corr_fp}개 ({low_corr_fp/len(fp_df)*100:.1f}%)")
        print(f"상관계수 0.4-0.5인 FP 쌍: {((fp_df['corr'].abs() >= 0.4) & (fp_df['corr'].abs() < 0.5)).sum()}개")
        print(f"상관계수 >= 0.5인 FP 쌍: {(fp_df['corr'].abs() >= 0.5).sum()}개")
    
    return fp_df


def test_threshold_optimization(pivot_train, pivot_for_answer):
    """상관계수 임계값 최적화 테스트"""
    print("\n" + "="*60)
    print("4. 상관계수 임계값 최적화")
    print("="*60)
    
    thresholds = [0.30, 0.32, 0.35, 0.37, 0.40, 0.42, 0.45]
    results = []
    
    answer_pairs = find_comovement_pairs(pivot_for_answer, max_lag=6, min_nonzero=12, corr_threshold=0.35)
    answer_pairs_set = set(zip(answer_pairs['leading_item_id'], answer_pairs['following_item_id']))
    
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
        
        print(f"임계값 {threshold:.2f}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, TP={tp}, FP={fp}, FN={fn}")
    
    results_df = pd.DataFrame(results)
    
    # 최적 임계값 찾기
    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_f1 = results_df.loc[best_idx, 'f1']
    
    print(f"\n최적 임계값: {best_threshold:.2f} (F1={best_f1:.4f})")
    
    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['threshold'], results_df['f1'], marker='o', label='F1 Score', linewidth=2)
    plt.plot(results_df['threshold'], results_df['precision'], marker='s', label='Precision', linewidth=2)
    plt.plot(results_df['threshold'], results_df['recall'], marker='^', label='Recall', linewidth=2)
    plt.axvline(best_threshold, color='r', linestyle='--', alpha=0.5, label=f'최적 임계값 ({best_threshold:.2f})')
    plt.xlabel('상관계수 임계값')
    plt.ylabel('점수')
    plt.title('상관계수 임계값에 따른 성능 변화')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../analysis/threshold_optimization.png', dpi=150, bbox_inches='tight')
    print("\n그래프 저장: analysis/threshold_optimization.png")
    
    return results_df, best_threshold


def main():
    """메인 실행 함수"""
    
    # 분석 결과 저장 디렉토리 생성
    os.makedirs('../analysis', exist_ok=True)
    
    print("="*60)
    print("공행성 쌍 탐색 개선 분석")
    print("="*60)
    
    # 데이터 로드 및 준비
    pivot = load_and_prepare_data()
    pivot_train, pivot_for_answer, split_idx, months = create_validation_split(pivot)
    
    # 1. 상관계수 분포 분석
    pred_pairs, answer_pairs, tp_pairs, fp_pairs, fn_pairs = analyze_correlation_distribution(
        pivot_train, pivot_for_answer, split_idx, months
    )
    
    # 2. Lag 분포 분석
    pred_dict, answer_dict = analyze_lag_distribution(pred_pairs, answer_pairs, tp_pairs, fp_pairs)
    
    # 3. FP 쌍 특성 분석
    fp_df = analyze_fp_characteristics(pivot_train, fp_pairs, pred_dict)
    if len(fp_df) > 0:
        fp_df.to_csv('../analysis/fp_pairs_analysis.csv', index=False)
        print("\nFP 쌍 분석 결과 저장: analysis/fp_pairs_analysis.csv")
    
    # 4. 임계값 최적화
    results_df, best_threshold = test_threshold_optimization(pivot_train, pivot_for_answer)
    results_df.to_csv('../analysis/threshold_optimization_results.csv', index=False)
    print("\n임계값 최적화 결과 저장: analysis/threshold_optimization_results.csv")
    
    print("\n" + "="*60)
    print("분석 완료!")
    print("="*60)
    print(f"\n주요 발견 사항:")
    print(f"- 최적 상관계수 임계값: {best_threshold:.2f}")
    print(f"- TP 평균 상관계수: {np.mean([pred_dict[pair][0] for pair in tp_pairs if pair in pred_dict]):.4f}")
    print(f"- FP 평균 상관계수: {np.mean([pred_dict[pair][0] for pair in fp_pairs if pair in pred_dict]):.4f}")


if __name__ == "__main__":
    main()

