"""
research3.txt의 인사이트를 바탕으로 새로운 피처 생성 아이디어 도출을 위한 EDA
- 품목별 통계량 기반 피처 (mean, std 등)
- 상호작용 변수 (lag × corr, 품목 특성 조합 등)
- Count encoding (품목별 등장 빈도 등)
- 시계열 통계량 확장 (더 긴 기간, 변동성 등)
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


def load_data():
    """데이터 로드 및 전처리"""
    print("=" * 60)
    print("[1단계] 데이터 로드 중...")
    print("=" * 60)
    
    train = pd.read_csv('../data/train.csv')
    print(f"학습 데이터 shape: {train.shape}")
    
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
    
    print(f"피벗 테이블 shape: {pivot.shape}")
    return train, pivot


def analyze_item_statistics(pivot):
    """품목별 통계량 분석 (research3.txt의 groupby 통계량 아이디어)"""
    print("\n" + "=" * 60)
    print("[2단계] 품목별 통계량 분석")
    print("=" * 60)
    
    # 품목별 통계량 계산
    item_stats = []
    for item_id in tqdm(pivot.index, desc="품목별 통계량 계산"):
        series = pivot.loc[item_id].values.astype(float)
        non_zero = series[series > 0]
        
        stats = {
            'item_id': item_id,
            'mean': np.mean(series),
            'std': np.std(series),
            'min': np.min(series),
            'max': np.max(series),
            'median': np.median(series),
            'non_zero_count': len(non_zero),
            'non_zero_mean': np.mean(non_zero) if len(non_zero) > 0 else 0,
            'non_zero_std': np.std(non_zero) if len(non_zero) > 0 else 0,
            'cv': np.std(series) / (np.mean(series) + 1e-6),  # 변동계수
            'zero_ratio': (len(series) - len(non_zero)) / len(series),
        }
        item_stats.append(stats)
    
    item_stats_df = pd.DataFrame(item_stats)
    
    print("\n품목별 통계량 요약:")
    print(item_stats_df.describe())
    
    # 통계량 분포 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('품목별 통계량 분포', fontsize=16, y=1.02)
    
    axes[0, 0].hist(item_stats_df['mean'], bins=50, edgecolor='black')
    axes[0, 0].set_title('평균 무역량 분포')
    axes[0, 0].set_xlabel('평균 무역량')
    axes[0, 0].set_ylabel('빈도')
    
    axes[0, 1].hist(item_stats_df['std'], bins=50, edgecolor='black')
    axes[0, 1].set_title('표준편차 분포')
    axes[0, 1].set_xlabel('표준편차')
    axes[0, 1].set_ylabel('빈도')
    
    axes[0, 2].hist(item_stats_df['cv'], bins=50, edgecolor='black')
    axes[0, 2].set_title('변동계수(CV) 분포')
    axes[0, 2].set_xlabel('변동계수')
    axes[0, 2].set_ylabel('빈도')
    
    axes[1, 0].hist(item_stats_df['zero_ratio'], bins=50, edgecolor='black')
    axes[1, 0].set_title('0 비율 분포')
    axes[1, 0].set_xlabel('0 비율')
    axes[1, 0].set_ylabel('빈도')
    
    axes[1, 1].scatter(item_stats_df['mean'], item_stats_df['std'], alpha=0.5)
    axes[1, 1].set_title('평균 vs 표준편차')
    axes[1, 1].set_xlabel('평균')
    axes[1, 1].set_ylabel('표준편차')
    
    axes[1, 2].scatter(item_stats_df['mean'], item_stats_df['cv'], alpha=0.5)
    axes[1, 2].set_title('평균 vs 변동계수')
    axes[1, 2].set_xlabel('평균')
    axes[1, 2].set_ylabel('변동계수')
    
    plt.tight_layout()
    plt.savefig('../analysis/item_statistics_distribution.png', dpi=300, bbox_inches='tight')
    print("\n품목별 통계량 분포 그래프 저장: ../analysis/item_statistics_distribution.png")
    
    return item_stats_df


def analyze_pair_interactions(pivot):
    """공행성 쌍의 상호작용 분석 (research3.txt의 상호작용 변수 아이디어)"""
    print("\n" + "=" * 60)
    print("[3단계] 공행성 쌍 상호작용 분석")
    print("=" * 60)
    
    # 간단한 공행성 쌍 탐색 (빠른 분석을 위해 샘플링)
    items = pivot.index.to_list()
    n_items = len(items)
    
    # 샘플링 (전체 조합이 너무 많으므로)
    sample_size = min(1000, n_items * (n_items - 1) // 2)
    np.random.seed(42)
    
    interactions = []
    sampled_pairs = []
    
    for _ in tqdm(range(sample_size), desc="쌍 상호작용 분석"):
        i, j = np.random.choice(n_items, 2, replace=False)
        leader = items[i]
        follower = items[j]
        
        if leader == follower:
            continue
        
        x = pivot.loc[leader].values.astype(float)
        y = pivot.loc[follower].values.astype(float)
        
        # lag별 상관계수 계산
        best_corr = 0.0
        best_lag = 0
        for lag in range(1, 7):
            if len(x) <= lag:
                continue
            corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        
        if abs(best_corr) >= 0.3:  # 임계값 이상인 쌍만
            interactions.append({
                'leading_item_id': leader,
                'following_item_id': follower,
                'max_corr': best_corr,
                'best_lag': best_lag,
                'leader_mean': np.mean(x),
                'leader_std': np.std(x),
                'follower_mean': np.mean(y),
                'follower_std': np.std(y),
            })
            sampled_pairs.append((leader, follower))
    
    interactions_df = pd.DataFrame(interactions)
    
    if len(interactions_df) > 0:
        print(f"\n발견된 공행성 쌍 수: {len(interactions_df)}")
        print("\n상호작용 통계:")
        print(interactions_df.describe())
        
        # 상호작용 변수 생성 아이디어
        interactions_df['lag_x_corr'] = interactions_df['best_lag'] * abs(interactions_df['max_corr'])
        interactions_df['mean_ratio'] = interactions_df['follower_mean'] / (interactions_df['leader_mean'] + 1e-6)
        interactions_df['std_ratio'] = interactions_df['follower_std'] / (interactions_df['leader_std'] + 1e-6)
        
        print("\n생성된 상호작용 변수:")
        print(interactions_df[['lag_x_corr', 'mean_ratio', 'std_ratio']].describe())
        
        # 상호작용 변수 분포 시각화
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('공행성 쌍 상호작용 변수 분포', fontsize=16)
        
        axes[0].hist(interactions_df['lag_x_corr'], bins=50, edgecolor='black')
        axes[0].set_title('lag × |corr| 분포')
        axes[0].set_xlabel('lag × |corr|')
        axes[0].set_ylabel('빈도')
        
        axes[1].hist(interactions_df['mean_ratio'], bins=50, edgecolor='black')
        axes[1].set_title('평균 비율 (follower/leader) 분포')
        axes[1].set_xlabel('평균 비율')
        axes[1].set_ylabel('빈도')
        
        axes[2].hist(interactions_df['std_ratio'], bins=50, edgecolor='black')
        axes[2].set_title('표준편차 비율 (follower/leader) 분포')
        axes[2].set_xlabel('표준편차 비율')
        axes[2].set_ylabel('빈도')
        
        plt.tight_layout()
        plt.savefig('../analysis/pair_interactions_distribution.png', dpi=300, bbox_inches='tight')
        print("\n상호작용 변수 분포 그래프 저장: ../analysis/pair_interactions_distribution.png")
    
    return interactions_df


def analyze_timeseries_features(pivot):
    """시계열 통계량 확장 분석 (research3.txt의 롤링 통계량 아이디어)"""
    print("\n" + "=" * 60)
    print("[4단계] 시계열 통계량 확장 분석")
    print("=" * 60)
    
    # 샘플 품목 선택
    sample_items = pivot.index[:10].tolist()
    
    timeseries_features = []
    
    for item_id in tqdm(sample_items, desc="시계열 통계량 계산"):
        series = pivot.loc[item_id].values.astype(float)
        
        # 다양한 기간의 이동평균 및 통계량
        for window in [3, 6, 12]:
            if len(series) >= window:
                ma = pd.Series(series).rolling(window=window, min_periods=1).mean()
                std = pd.Series(series).rolling(window=window, min_periods=1).std().fillna(0)
                max_val = pd.Series(series).rolling(window=window, min_periods=1).max()
                min_val = pd.Series(series).rolling(window=window, min_periods=1).min()
                
                # 마지막 값들만 저장
                timeseries_features.append({
                    'item_id': item_id,
                    'window': window,
                    'ma_mean': ma.iloc[-1],
                    'ma_std': std.iloc[-1],
                    'ma_max': max_val.iloc[-1],
                    'ma_min': min_val.iloc[-1],
                    'ma_range': max_val.iloc[-1] - min_val.iloc[-1],
                })
    
    ts_features_df = pd.DataFrame(timeseries_features)
    
    if len(ts_features_df) > 0:
        print("\n시계열 통계량 요약:")
        print(ts_features_df.groupby('window').describe())
        
        # 시계열 통계량 비교 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('시계열 통계량 비교 (기간별)', fontsize=16)
        
        for window in [3, 6, 12]:
            window_data = ts_features_df[ts_features_df['window'] == window]
            if len(window_data) > 0:
                axes[0, 0].hist(window_data['ma_mean'], bins=20, alpha=0.6, label=f'{window}개월', edgecolor='black')
                axes[0, 1].hist(window_data['ma_std'], bins=20, alpha=0.6, label=f'{window}개월', edgecolor='black')
                axes[1, 0].hist(window_data['ma_range'], bins=20, alpha=0.6, label=f'{window}개월', edgecolor='black')
        
        axes[0, 0].set_title('이동평균 분포')
        axes[0, 0].set_xlabel('이동평균')
        axes[0, 0].set_ylabel('빈도')
        axes[0, 0].legend()
        
        axes[0, 1].set_title('이동표준편차 분포')
        axes[0, 1].set_xlabel('이동표준편차')
        axes[0, 1].set_ylabel('빈도')
        axes[0, 1].legend()
        
        axes[1, 0].set_title('이동범위 분포')
        axes[1, 0].set_xlabel('이동범위')
        axes[1, 0].set_ylabel('빈도')
        axes[1, 0].legend()
        
        # 기간별 통계량 비교
        window_comparison = ts_features_df.groupby('window')['ma_mean'].mean()
        axes[1, 1].bar(window_comparison.index, window_comparison.values, edgecolor='black')
        axes[1, 1].set_title('기간별 평균 이동평균')
        axes[1, 1].set_xlabel('기간(개월)')
        axes[1, 1].set_ylabel('평균 이동평균')
        
        plt.tight_layout()
        plt.savefig('../analysis/timeseries_features_comparison.png', dpi=300, bbox_inches='tight')
        print("\n시계열 통계량 비교 그래프 저장: ../analysis/timeseries_features_comparison.png")
    
    return ts_features_df


def analyze_count_encoding(pivot, train):
    """Count encoding 분석 (research3.txt의 count encoding 아이디어)"""
    print("\n" + "=" * 60)
    print("[5단계] Count Encoding 분석")
    print("=" * 60)
    
    # 품목별 등장 횟수 (월별 데이터가 있는 경우)
    item_counts = train.groupby('item_id').size().reset_index(name='total_count')
    item_counts['monthly_count'] = train.groupby('item_id')['month'].nunique().reset_index(name='monthly_count')['monthly_count']
    
    # 품목별 평균 무역량이 0이 아닌 월의 개수
    item_nonzero_counts = []
    for item_id in tqdm(pivot.index, desc="품목별 0이 아닌 월 개수 계산"):
        series = pivot.loc[item_id].values.astype(float)
        item_nonzero_counts.append({
            'item_id': item_id,
            'nonzero_month_count': np.count_nonzero(series),
            'total_month_count': len(series),
        })
    
    count_df = pd.DataFrame(item_nonzero_counts)
    count_df = count_df.merge(item_counts, on='item_id', how='left')
    count_df['nonzero_ratio'] = count_df['nonzero_month_count'] / count_df['total_month_count']
    
    print("\nCount Encoding 통계:")
    print(count_df.describe())
    
    # Count encoding 분포 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Count Encoding 분포', fontsize=16)
    
    axes[0, 0].hist(count_df['total_count'], bins=50, edgecolor='black')
    axes[0, 0].set_title('품목별 총 등장 횟수')
    axes[0, 0].set_xlabel('총 등장 횟수')
    axes[0, 0].set_ylabel('빈도')
    
    axes[0, 1].hist(count_df['monthly_count'], bins=50, edgecolor='black')
    axes[0, 1].set_title('품목별 월별 등장 횟수')
    axes[0, 1].set_xlabel('월별 등장 횟수')
    axes[0, 1].set_ylabel('빈도')
    
    axes[1, 0].hist(count_df['nonzero_month_count'], bins=50, edgecolor='black')
    axes[1, 0].set_title('품목별 0이 아닌 월 개수')
    axes[1, 0].set_xlabel('0이 아닌 월 개수')
    axes[1, 0].set_ylabel('빈도')
    
    axes[1, 1].hist(count_df['nonzero_ratio'], bins=50, edgecolor='black')
    axes[1, 1].set_title('품목별 0이 아닌 월 비율')
    axes[1, 1].set_xlabel('0이 아닌 월 비율')
    axes[1, 1].set_ylabel('빈도')
    
    plt.tight_layout()
    plt.savefig('../analysis/count_encoding_distribution.png', dpi=300, bbox_inches='tight')
    print("\nCount Encoding 분포 그래프 저장: ../analysis/count_encoding_distribution.png")
    
    return count_df


def generate_feature_suggestions(item_stats_df, interactions_df, ts_features_df, count_df):
    """분석 결과를 바탕으로 새로운 피처 생성 제안"""
    print("\n" + "=" * 60)
    print("[6단계] 새로운 피처 생성 제안")
    print("=" * 60)
    
    suggestions = []
    
    # 1. 품목별 통계량 기반 피처
    suggestions.append({
        'category': '품목별 통계량',
        'feature_name': 'item_mean',
        'description': '품목별 전체 기간 평균 무역량',
        'implementation': 'item_stats_df의 mean 컬럼 사용',
        'priority': '높음'
    })
    suggestions.append({
        'category': '품목별 통계량',
        'feature_name': 'item_std',
        'description': '품목별 전체 기간 표준편차',
        'implementation': 'item_stats_df의 std 컬럼 사용',
        'priority': '높음'
    })
    suggestions.append({
        'category': '품목별 통계량',
        'feature_name': 'item_cv',
        'description': '품목별 변동계수 (std/mean)',
        'implementation': 'item_stats_df의 cv 컬럼 사용',
        'priority': '중간'
    })
    suggestions.append({
        'category': '품목별 통계량',
        'feature_name': 'item_zero_ratio',
        'description': '품목별 0인 월의 비율',
        'implementation': 'item_stats_df의 zero_ratio 컬럼 사용',
        'priority': '중간'
    })
    
    # 2. 상호작용 변수
    if len(interactions_df) > 0:
        suggestions.append({
            'category': '상호작용 변수',
            'feature_name': 'lag_x_corr',
            'description': 'lag와 상관계수의 곱 (관계 강도 × 시차)',
            'implementation': 'best_lag * abs(max_corr)',
            'priority': '높음'
        })
        suggestions.append({
            'category': '상호작용 변수',
            'feature_name': 'mean_ratio',
            'description': '후행 품목 평균 / 선행 품목 평균',
            'implementation': 'follower_mean / (leader_mean + 1e-6)',
            'priority': '중간'
        })
        suggestions.append({
            'category': '상호작용 변수',
            'feature_name': 'std_ratio',
            'description': '후행 품목 표준편차 / 선행 품목 표준편차',
            'implementation': 'follower_std / (leader_std + 1e-6)',
            'priority': '중간'
        })
    
    # 3. 시계열 통계량 확장
    suggestions.append({
        'category': '시계열 통계량',
        'feature_name': 'b_ma6',
        'description': '후행 품목의 6개월 이동평균',
        'implementation': 'pd.Series(b_series).rolling(6).mean()',
        'priority': '높음'
    })
    suggestions.append({
        'category': '시계열 통계량',
        'feature_name': 'b_ma12',
        'description': '후행 품목의 12개월 이동평균',
        'implementation': 'pd.Series(b_series).rolling(12).mean()',
        'priority': '중간'
    })
    suggestions.append({
        'category': '시계열 통계량',
        'feature_name': 'b_std3',
        'description': '후행 품목의 3개월 이동표준편차',
        'implementation': 'pd.Series(b_series).rolling(3).std()',
        'priority': '높음'
    })
    suggestions.append({
        'category': '시계열 통계량',
        'feature_name': 'b_range3',
        'description': '후행 품목의 3개월 범위 (max - min)',
        'implementation': 'pd.Series(b_series).rolling(3).max() - pd.Series(b_series).rolling(3).min()',
        'priority': '중간'
    })
    
    # 4. Count encoding
    suggestions.append({
        'category': 'Count Encoding',
        'feature_name': 'leader_nonzero_count',
        'description': '선행 품목의 0이 아닌 월 개수',
        'implementation': 'count_df의 nonzero_month_count 사용',
        'priority': '중간'
    })
    suggestions.append({
        'category': 'Count Encoding',
        'feature_name': 'follower_nonzero_count',
        'description': '후행 품목의 0이 아닌 월 개수',
        'implementation': 'count_df의 nonzero_month_count 사용',
        'priority': '중간'
    })
    
    suggestions_df = pd.DataFrame(suggestions)
    
    print("\n새로운 피처 생성 제안:")
    print(suggestions_df.to_string(index=False))
    
    # CSV로 저장
    suggestions_df.to_csv('../analysis/feature_suggestions.csv', index=False, encoding='utf-8-sig')
    print(f"\n피처 제안 저장: ../analysis/feature_suggestions.csv")
    
    return suggestions_df


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("research3.txt 인사이트 기반 새로운 피처 생성 EDA")
    print("=" * 60)
    
    # 데이터 로드
    train, pivot = load_data()
    
    # 1. 품목별 통계량 분석
    item_stats_df = analyze_item_statistics(pivot)
    item_stats_df.to_csv('../analysis/item_statistics.csv', index=False, encoding='utf-8-sig')
    
    # 2. 공행성 쌍 상호작용 분석
    interactions_df = analyze_pair_interactions(pivot)
    if len(interactions_df) > 0:
        interactions_df.to_csv('../analysis/pair_interactions.csv', index=False, encoding='utf-8-sig')
    
    # 3. 시계열 통계량 확장 분석
    ts_features_df = analyze_timeseries_features(pivot)
    if len(ts_features_df) > 0:
        ts_features_df.to_csv('../analysis/timeseries_features.csv', index=False, encoding='utf-8-sig')
    
    # 4. Count encoding 분석
    count_df = analyze_count_encoding(pivot, train)
    count_df.to_csv('../analysis/count_encoding.csv', index=False, encoding='utf-8-sig')
    
    # 5. 피처 생성 제안
    suggestions_df = generate_feature_suggestions(item_stats_df, interactions_df, ts_features_df, count_df)
    
    print("\n" + "=" * 60)
    print("EDA 완료!")
    print("=" * 60)
    print("\n생성된 파일:")
    print("  - ../analysis/item_statistics.csv")
    print("  - ../analysis/pair_interactions.csv")
    print("  - ../analysis/timeseries_features.csv")
    print("  - ../analysis/count_encoding.csv")
    print("  - ../analysis/feature_suggestions.csv")
    print("  - ../analysis/*.png (시각화 그래프)")


if __name__ == "__main__":
    main()

