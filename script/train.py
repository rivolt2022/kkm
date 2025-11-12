"""
국민대학교 AI빅데이터 분석 경진대회 - 공행성 쌍 판별 및 무역량 예측
Baseline 개선 버전: XGBoost 회귀 모델 및 향상된 feature engineering 사용
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def safe_corr(x, y):
    """안전한 상관계수 계산 (표준편차가 0인 경우 처리)"""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def find_comovement_pairs(pivot, max_lag=6, min_nonzero=12, corr_threshold=0.35):
    """
    공행성 쌍 탐색
    - 각 (A, B) 쌍에 대해 lag = 1 ~ max_lag까지 Pearson 상관계수 계산
    - 절댓값이 가장 큰 상관계수와 lag를 선택
    - |corr| >= corr_threshold이면 A→B 공행성 있다고 판단
    """
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)

    results = []

    for i, leader in tqdm(enumerate(items), total=len(items), desc="공행성 쌍 탐색"):
        x = pivot.loc[leader].values.astype(float)
        if np.count_nonzero(x) < min_nonzero:
            continue

        for follower in items:
            if follower == leader:
                continue

            y = pivot.loc[follower].values.astype(float)
            if np.count_nonzero(y) < min_nonzero:
                continue

            best_lag = None
            best_corr = 0.0

            # lag = 1 ~ max_lag 탐색
            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue
                corr = safe_corr(x[:-lag], y[lag:])
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag


            # 임계값 이상이면 공행성쌍으로 채택
            if best_lag is not None and abs(best_corr) >= corr_threshold:
                results.append({
                    "leading_item_id": leader,
                    "following_item_id": follower,
                    "best_lag": best_lag,
                    "max_corr": best_corr,
                })

    pairs = pd.DataFrame(results)
    return pairs


def build_training_data(pivot, pairs):
    """
    공행성쌍 + 시계열을 이용해 (X, y) 학습 데이터를 만드는 함수
    향상된 feature engineering:
    - b_t, b_t_1: 후행 품목의 현재 및 직전 달 무역량
    - a_t_lag: 선행 품목의 lag 반영된 무역량
    - max_corr, best_lag: 관계 특성
    - b_trend: 후행 품목의 추세 (최근 3개월 평균)
    - a_trend: 선행 품목의 추세 (최근 3개월 평균)
    - b_ma3: 후행 품목의 3개월 이동평균
    """
    months = pivot.columns.to_list()
    n_months = len(months)

    rows = []

    for row in tqdm(pairs.itertuples(index=False), total=len(pairs), desc="학습 데이터 생성"):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot.index or follower not in pivot.index:
            continue

        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)

        # t+1이 존재하고, t-lag >= 0인 구간만 학습에 사용
        for t in range(max(lag, 2), n_months - 1):
            b_t = b_series[t]
            b_t_1 = b_series[t - 1]
            a_t_lag = a_series[t - lag]
            b_t_plus_1 = b_series[t + 1]

            # 추가 feature 계산
            # 추세: 최근 3개월 평균
            if t >= 2:
                b_trend = np.mean(b_series[max(0, t-2):t+1]) if t >= 2 else b_t
                a_trend = np.mean(a_series[max(0, t-lag-2):t-lag+1]) if t-lag >= 2 else a_t_lag
            else:
                b_trend = b_t
                a_trend = a_t_lag

            # 이동평균
            if t >= 2:
                b_ma3 = np.mean(b_series[max(0, t-2):t+1])
            else:
                b_ma3 = b_t

            # 변화율
            if b_t_1 > 0:
                b_change = (b_t - b_t_1) / (b_t_1 + 1e-6)
            else:
                b_change = 0.0

            rows.append({
                "b_t": b_t,
                "b_t_1": b_t_1,
                "a_t_lag": a_t_lag,
                "max_corr": corr,
                "best_lag": float(lag),
                "b_trend": b_trend,
                "a_trend": a_trend,
                "b_ma3": b_ma3,
                "b_change": b_change,
                "target": b_t_plus_1,
            })

    df_train = pd.DataFrame(rows)
    return df_train


def predict(pivot, pairs, reg, feature_cols):
    """
    회귀 모델 추론 및 제출 파일 생성
    탐색된 공행성 쌍에 대해 후행 품목의 2025년 8월 총 무역량 예측
    """
    months = pivot.columns.to_list()
    n_months = len(months)

    # 가장 마지막 두 달 index (2025-7, 2025-6)
    t_last = n_months - 1
    t_prev = n_months - 2

    preds = []

    for row in tqdm(pairs.itertuples(index=False), total=len(pairs), desc="예측 수행"):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot.index or follower not in pivot.index:
            continue

        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)

        # t_last - lag 가 0 이상인 경우만 예측
        if t_last - lag < 0:
            continue

        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]

        # 추가 feature 계산
        if t_last >= 2:
            b_trend = np.mean(b_series[max(0, t_last-2):t_last+1])
            a_trend = np.mean(a_series[max(0, t_last-lag-2):t_last-lag+1]) if t_last-lag >= 2 else a_t_lag
        else:
            b_trend = b_t
            a_trend = a_t_lag

        if t_last >= 2:
            b_ma3 = np.mean(b_series[max(0, t_last-2):t_last+1])
        else:
            b_ma3 = b_t

        if b_t_1 > 0:
            b_change = (b_t - b_t_1) / (b_t_1 + 1e-6)
        else:
            b_change = 0.0

        # Feature 벡터 구성
        features = {
            "b_t": b_t,
            "b_t_1": b_t_1,
            "a_t_lag": a_t_lag,
            "max_corr": corr,
            "best_lag": float(lag),
            "b_trend": b_trend,
            "a_trend": a_trend,
            "b_ma3": b_ma3,
            "b_change": b_change,
        }

        X_test = np.array([[features[col] for col in feature_cols]])
        y_pred = reg.predict(X_test)[0]

        # 후처리: 음수 예측 → 0으로 변환, 소수점 → 정수 변환
        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("국민대학교 AI빅데이터 분석 경진대회 - 학습 및 예측")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1단계] 데이터 로드 중...")
    train = pd.read_csv('../data/train.csv')
    print(f"학습 데이터 shape: {train.shape}")

    # 2. 데이터 전처리
    print("\n[2단계] 데이터 전처리 중...")
    # year, month, item_id 기준으로 value 합산
    monthly = (
        train
        .groupby(["item_id", "year", "month"], as_index=False)["value"]
        .sum()
    )

    # year, month를 하나의 키(ym)로 묶기
    monthly["ym"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )

    # item_id × ym 피벗 (월별 총 무역량 매트릭스 생성)
    pivot = (
        monthly
        .pivot(index="item_id", columns="ym", values="value")
        .fillna(0.0)
    )
    print(f"피벗 테이블 shape: {pivot.shape}")

    # 3. 공행성쌍 탐색
    print("\n[3단계] 공행성 쌍 탐색 중...")
    pairs = find_comovement_pairs(pivot, max_lag=6, min_nonzero=12, corr_threshold=0.35)
    print(f"탐색된 공행성쌍 수: {len(pairs)}")

    if len(pairs) == 0:
        print("경고: 공행성 쌍이 발견되지 않았습니다.")
        return

    # 4. 학습 데이터 생성
    print("\n[4단계] 학습 데이터 생성 중...")
    df_train_model = build_training_data(pivot, pairs)
    print(f'생성된 학습 데이터 shape: {df_train_model.shape}')

    # 5. 회귀 모델 학습
    print("\n[5단계] 회귀 모델 학습 중...")
    feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                    'b_trend', 'a_trend', 'b_ma3', 'b_change']

    train_X = df_train_model[feature_cols].values
    train_y = df_train_model["target"].values

    # XGBoost 회귀 모델 사용 (성능 향상)
    print("XGBoost 회귀 모델 학습 중...")
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
    print("모델 학습 완료")

    # 6. 예측 및 제출 파일 생성
    print("\n[6단계] 예측 수행 중...")
    submission = predict(pivot, pairs, reg, feature_cols)
    print(f"예측된 쌍 수: {len(submission)}")

    # 7. 제출 파일 저장
    print("\n[7단계] 제출 파일 저장 중...")
    output_path = '../data/submission.csv'
    submission.to_csv(output_path, index=False)
    print(f"제출 파일 저장 완료: {output_path}")
    print(f"\n제출 파일 미리보기:")
    print(submission.head(10))
    print(f"\n총 {len(submission)}개의 공행성 쌍에 대한 예측이 완료되었습니다.")


if __name__ == "__main__":
    main()

