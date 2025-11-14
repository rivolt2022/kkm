"""
Feature Engineering 개선 분석 스크립트
- 예측 오차 패턴 분석
- 시계열 특성 분석 (계절성, 트렌드, 변동성)
- 스케일 문제 분석
- 추가 feature 아이디어 도출
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import warnings
from tqdm import tqdm
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings('ignore')

# train.py의 함수들을 재사용
sys.path.append(os.path.dirname(__file__))
from train import (
    find_comovement_pairs, build_training_data, predict,
    safe_corr
)

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


def analyze_prediction_errors(pivot_train, pivot_for_answer, split_idx, months):
    """예측 오차 패턴 분석"""
    print("\n" + "="*60)
    print("1. 예측 오차 패턴 분석")
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
    
    # 오차 계산
    pred_dict = dict(zip(
        zip(submission['leading_item_id'], submission['following_item_id']),
        submission['value']
    ))
    answer_dict = dict(zip(
        zip(answer_df['leading_item_id'], answer_df['following_item_id']),
        answer_df['value']
    ))
    
    errors = []
    for pair in set(list(pred_dict.keys()) + list(answer_dict.keys())):
        if pair in pred_dict and pair in answer_dict:
            y_true = answer_dict[pair]
            y_pred = pred_dict[pair]
            if y_true > 0:
                rel_error = abs(y_true - y_pred) / y_true
                rel_error = min(rel_error, 1.0)  # 100% 이상은 100%로
            else:
                rel_error = 1.0 if y_pred > 0 else 0.0
            
            errors.append({
                'pair': pair,
                'y_true': y_true,
                'y_pred': y_pred,
                'abs_error': abs(y_true - y_pred),
                'rel_error': rel_error,
                'is_overestimate': y_pred > y_true,
            })
    
    errors_df = pd.DataFrame(errors)
    
    if len(errors_df) > 0:
        print(f"\n오차 통계:")
        print(f"평균 상대 오차: {errors_df['rel_error'].mean():.4f}")
        print(f"중앙값 상대 오차: {errors_df['rel_error'].median():.4f}")
        print(f"과대 예측 비율: {errors_df['is_overestimate'].mean():.4f}")
        
        # 오차 분포 시각화
        plt.figure(figsize=(12, 6))
        plt.hist(errors_df['rel_error'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('상대 오차')
        plt.ylabel('빈도')
        plt.title('예측 오차 분포')
        plt.axvline(errors_df['rel_error'].mean(), color='r', linestyle='--', label=f'평균: {errors_df["rel_error"].mean():.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('../analysis/prediction_error_distribution.png', dpi=150, bbox_inches='tight')
        print("\n그래프 저장: analysis/prediction_error_distribution.png")
        
        errors_df.to_csv('../analysis/prediction_errors.csv', index=False)
        print("오차 데이터 저장: analysis/prediction_errors.csv")
    
    return errors_df


def analyze_scale_issues(df_train):
    """스케일 문제 분석"""
    print("\n" + "="*60)
    print("2. 스케일 문제 분석")
    print("="*60)
    
    feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                    'b_trend', 'a_trend', 'b_ma3', 'b_change']
    
    scale_stats = {}
    for col in feature_cols:
        if col in df_train.columns:
            values = df_train[col].values
            scale_stats[col] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf,
            }
    
    scale_df = pd.DataFrame(scale_stats).T
    print("\nFeature 스케일 통계:")
    print(scale_df)
    
    scale_df.to_csv('../analysis/feature_scale_stats.csv')
    print("\n스케일 통계 저장: analysis/feature_scale_stats.csv")
    
    # 스케일 차이 시각화
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(feature_cols):
        if col in df_train.columns:
            plt.subplot(3, 3, i+1)
            values = df_train[col].values
            # 이상치 제거 (상위 1%)
            q99 = np.percentile(values, 99)
            values_filtered = values[values <= q99]
            plt.hist(values_filtered, bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'{col}\n(mean={np.mean(values):.2e}, std={np.std(values):.2e})')
            plt.xlabel('값')
            plt.ylabel('빈도')
    
    plt.tight_layout()
    plt.savefig('../analysis/feature_scale_distributions.png', dpi=150, bbox_inches='tight')
    print("\n그래프 저장: analysis/feature_scale_distributions.png")
    
    return scale_df


def analyze_time_series_characteristics(pivot_train):
    """시계열 특성 분석"""
    print("\n" + "="*60)
    print("3. 시계열 특성 분석")
    print("="*60)
    
    # 몇 개 품목 샘플링
    sample_items = pivot_train.index[:10].tolist()
    
    ts_characteristics = []
    
    for item_id in sample_items:
        series = pivot_train.loc[item_id].values.astype(float)
        non_zero_series = series[series > 0]
        
        if len(non_zero_series) < 5:
            continue
        
        # 변동성
        cv = np.std(non_zero_series) / np.mean(non_zero_series) if np.mean(non_zero_series) > 0 else 0
        
        # 트렌드 (선형 회귀 기울기)
        x = np.arange(len(non_zero_series))
        if len(x) > 1:
            trend = np.polyfit(x, non_zero_series, 1)[0]
        else:
            trend = 0
        
        # 계절성 (월별 패턴) - 간단히 분산으로 측정
        monthly_values = []
        for i in range(0, len(series), 12):
            if i + 12 <= len(series):
                monthly_values.append(series[i:i+12])
        
        if len(monthly_values) > 0:
            seasonal_var = np.var([np.mean(m) for m in monthly_values])
        else:
            seasonal_var = 0
        
        ts_characteristics.append({
            'item_id': item_id,
            'mean': np.mean(non_zero_series),
            'std': np.std(non_zero_series),
            'cv': cv,
            'trend': trend,
            'seasonal_var': seasonal_var,
            'zero_ratio': (series == 0).sum() / len(series),
        })
    
    ts_df = pd.DataFrame(ts_characteristics)
    
    if len(ts_df) > 0:
        print("\n시계열 특성 통계:")
        print(f"평균 변동계수: {ts_df['cv'].mean():.4f}")
        print(f"평균 0 비율: {ts_df['zero_ratio'].mean():.4f}")
        print(f"평균 트렌드: {ts_df['trend'].mean():.2e}")
        
        ts_df.to_csv('../analysis/time_series_characteristics.csv', index=False)
        print("\n시계열 특성 저장: analysis/time_series_characteristics.csv")
    
    return ts_df


def suggest_new_features(pivot_train, pairs, df_train):
    """새로운 feature 아이디어 제안"""
    print("\n" + "="*60)
    print("4. 새로운 Feature 아이디어")
    print("="*60)
    
    suggestions = []
    
    # 1. 계절성 feature (월별 패턴)
    suggestions.append({
        'feature_name': 'month_seasonality',
        'description': '월별 계절성 지표 (1-12월 패턴)',
        'implementation': '각 월의 평균 무역량을 feature로 추가',
        'expected_benefit': '계절적 패턴을 반영하여 예측 정확도 향상'
    })
    
    # 2. 변동성 feature
    suggestions.append({
        'feature_name': 'volatility',
        'description': '최근 N개월의 변동성 (표준편차/평균)',
        'implementation': '최근 6개월의 CV 계산',
        'expected_benefit': '변동성이 큰 품목의 예측 개선'
    })
    
    # 3. 상대적 크기 feature
    suggestions.append({
        'feature_name': 'relative_size',
        'description': '선행/후행 품목의 상대적 크기 비율',
        'implementation': 'a_t_lag / b_t (스케일 차이 보정)',
        'expected_benefit': '스케일 차이 문제 완화'
    })
    
    # 4. 누적 합계 feature
    suggestions.append({
        'feature_name': 'cumulative_sum',
        'description': '최근 N개월 누적 합계',
        'implementation': '최근 3개월, 6개월 누적 합계',
        'expected_benefit': '장기 트렌드 반영'
    })
    
    # 5. 변화율 가속도
    suggestions.append({
        'feature_name': 'acceleration',
        'description': '변화율의 변화 (2차 미분)',
        'implementation': '(b_t - b_t_1) - (b_t_1 - b_t_2)',
        'expected_benefit': '가속/감속 패턴 반영'
    })
    
    suggestions_df = pd.DataFrame(suggestions)
    suggestions_df.to_csv('../analysis/feature_suggestions.csv', index=False)
    
    print("\n제안된 Feature:")
    for idx, row in suggestions_df.iterrows():
        print(f"\n{idx+1}. {row['feature_name']}")
        print(f"   설명: {row['description']}")
        print(f"   구현: {row['implementation']}")
        print(f"   기대 효과: {row['expected_benefit']}")
    
    print("\nFeature 제안 저장: analysis/feature_suggestions.csv")
    
    return suggestions_df


def main():
    """메인 실행 함수"""
    # 분석 결과 저장 디렉토리 생성
    os.makedirs('../analysis', exist_ok=True)
    
    print("="*60)
    print("Feature Engineering 개선 분석")
    print("="*60)
    
    # 데이터 로드 및 준비
    pivot = load_and_prepare_data()
    pivot_train, pivot_for_answer, split_idx, months = create_validation_split(pivot)
    
    # 공행성 쌍 탐색
    pairs = find_comovement_pairs(pivot_train, max_lag=6, min_nonzero=12, corr_threshold=0.35)
    
    # 학습 데이터 생성
    df_train = build_training_data(pivot_train, pairs)
    
    # 1. 예측 오차 패턴 분석
    errors_df = analyze_prediction_errors(pivot_train, pivot_for_answer, split_idx, months)
    
    # 2. 스케일 문제 분석
    scale_df = analyze_scale_issues(df_train)
    
    # 3. 시계열 특성 분석
    ts_df = analyze_time_series_characteristics(pivot_train)
    
    # 4. 새로운 feature 제안
    suggestions_df = suggest_new_features(pivot_train, pairs, df_train)
    
    print("\n" + "="*60)
    print("분석 완료!")
    print("="*60)


if __name__ == "__main__":
    main()

