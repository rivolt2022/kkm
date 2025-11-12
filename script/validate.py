"""
국민대학교 AI빅데이터 분석 경진대회 - Public Score 예측 스크립트
train.csv의 일부 데이터를 검증 데이터로 분리하여 점수를 예측합니다.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
import warnings
import sys
import os

# evaluation.py 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../document'))
from evaluation import comovement_score, comovement_f1, comovement_nmae

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
    회귀 모델 추론
    pivot의 마지막 달 다음 달을 예측합니다.
    """
    months = pivot.columns.to_list()
    n_months = len(months)

    # 예측 시점: pivot의 마지막 달 (다음 달을 예측하기 위한 기준 시점)
    t_last = n_months - 1
    t_prev = n_months - 2

    if t_last < 0 or t_prev < 0:
        print(f"경고: 예측할 수 없는 시점입니다. t_last={t_last}, t_prev={t_prev}")
        return pd.DataFrame()

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
        b_t_1 = b_series[t_prev] if t_prev >= 0 else b_t
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


def create_answer_from_validation_data(validation_data, target_year, target_month, pairs):
    """
    검증 데이터에서 정답 파일 생성
    target_year, target_month: 예측 대상 달 (예: 2025, 7)
    pairs: 학습 데이터에서 찾은 공행성 쌍 (이 쌍들에 대해 정답 생성)
    """
    # 검증 데이터에서 target_year, target_month의 실제 무역량 추출
    validation_monthly = (
        validation_data
        .groupby(["item_id", "year", "month"], as_index=False)["value"]
        .sum()
    )
    
    # target_year, target_month에 해당하는 데이터만 필터링
    target_data = validation_monthly[
        (validation_monthly["year"] == target_year) & 
        (validation_monthly["month"] == target_month)
    ].copy()
    
    if len(target_data) == 0:
        print(f"경고: {target_year}년 {target_month}월 데이터가 없습니다.")
        return pd.DataFrame(columns=["leading_item_id", "following_item_id", "value"])
    
    # item_id -> value 매핑 딕셔너리
    value_dict = dict(zip(target_data["item_id"], target_data["value"]))
    
    # 학습 데이터에서 찾은 공행성 쌍에 대해 정답 생성
    answer_list = []
    for _, row in pairs.iterrows():
        leader = row["leading_item_id"]
        follower = row["following_item_id"]
        
        # 후행 품목의 실제 무역량이 있는 경우만 포함
        if follower in value_dict:
            answer_list.append({
                "leading_item_id": leader,
                "following_item_id": follower,
                "value": int(round(value_dict[follower]))
            })
    
    answer_df = pd.DataFrame(answer_list)
    return answer_df


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("국민대학교 AI빅데이터 분석 경진대회 - Public Score 예측")
    print("=" * 60)
    
    # 검증 설정
    # 마지막 N개월을 검증 데이터로 사용
    VALIDATION_MONTHS = 1  # 마지막 1개월을 검증 데이터로 사용
    
    print(f"\n검증 설정: 마지막 {VALIDATION_MONTHS}개월을 검증 데이터로 사용")
    
    # 1. 데이터 로드
    print("\n[1단계] 데이터 로드 중...")
    train = pd.read_csv('../data/train.csv')
    print(f"전체 학습 데이터 shape: {train.shape}")
    
    # 2. 데이터 분리 (학습/검증)
    print("\n[2단계] 데이터 분리 중...")
    # year, month 기준으로 정렬
    train_sorted = train.sort_values(['year', 'month'])
    
    # 고유한 (year, month) 조합 찾기
    train_sorted['ym'] = pd.to_datetime(
        train_sorted['year'].astype(str) + '-' + train_sorted['month'].astype(str).str.zfill(2)
    )
    unique_ym = sorted(train_sorted['ym'].unique())
    
    print(f"전체 기간: {unique_ym[0].strftime('%Y-%m')} ~ {unique_ym[-1].strftime('%Y-%m')}")
    print(f"총 {len(unique_ym)}개월 데이터")
    
    # 마지막 VALIDATION_MONTHS개월을 검증 데이터로 분리
    split_idx = len(unique_ym) - VALIDATION_MONTHS
    train_ym = unique_ym[:split_idx]
    validation_ym = unique_ym[split_idx:]
    
    print(f"\n학습 기간: {train_ym[0].strftime('%Y-%m')} ~ {train_ym[-1].strftime('%Y-%m')} ({len(train_ym)}개월)")
    print(f"검증 기간: {validation_ym[0].strftime('%Y-%m')} ~ {validation_ym[-1].strftime('%Y-%m')} ({len(validation_ym)}개월)")
    
    # 데이터 분리
    train_data = train_sorted[train_sorted['ym'].isin(train_ym)].copy()
    validation_data = train_sorted[train_sorted['ym'].isin(validation_ym)].copy()
    
    print(f"학습 데이터 shape: {train_data.shape}")
    print(f"검증 데이터 shape: {validation_data.shape}")
    
    # 예측 대상: 검증 기간의 마지막 달
    target_ym = validation_ym[-1]
    target_year = target_ym.year
    target_month = target_ym.month
    
    print(f"\n예측 대상: {target_year}년 {target_month}월")
    
    # 3. 학습 데이터 전처리
    print("\n[3단계] 학습 데이터 전처리 중...")
    monthly = (
        train_data
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
    
    # 4. 공행성쌍 탐색
    print("\n[4단계] 공행성 쌍 탐색 중...")
    pairs = find_comovement_pairs(pivot, max_lag=6, min_nonzero=12, corr_threshold=0.35)
    print(f"탐색된 공행성쌍 수: {len(pairs)}")
    
    if len(pairs) == 0:
        print("경고: 공행성 쌍이 발견되지 않았습니다.")
        return
    
    # 5. 학습 데이터 생성
    print("\n[5단계] 학습 데이터 생성 중...")
    df_train_model = build_training_data(pivot, pairs)
    print(f'생성된 학습 데이터 shape: {df_train_model.shape}')
    
    if len(df_train_model) == 0:
        print("경고: 학습 데이터가 생성되지 않았습니다.")
        return
    
    # 6. 회귀 모델 학습
    print("\n[6단계] 회귀 모델 학습 중...")
    feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'max_corr', 'best_lag', 
                    'b_trend', 'a_trend', 'b_ma3', 'b_change']
    
    train_X = df_train_model[feature_cols].values
    train_y = df_train_model["target"].values
    
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
    
    # 7. 예측 수행
    print("\n[7단계] 예측 수행 중...")
    # pivot의 마지막 달 다음 달을 예측 (검증 데이터의 마지막 달)
    submission = predict(pivot, pairs, reg, feature_cols)
    print(f"예측된 쌍 수: {len(submission)}")
    
    if len(submission) == 0:
        print("경고: 예측된 쌍이 없습니다.")
        return
    
    # 8. 정답 파일 생성
    print("\n[8단계] 정답 파일 생성 중...")
    answer = create_answer_from_validation_data(validation_data, target_year, target_month, pairs)
    print(f"정답 쌍 수: {len(answer)}")
    
    if len(answer) == 0:
        print("경고: 정답 파일이 생성되지 않았습니다.")
        return
    
    # 9. 점수 계산
    print("\n[9단계] 점수 계산 중...")
    print("=" * 60)
    
    try:
        f1 = comovement_f1(answer, submission)
        nmae = comovement_nmae(answer, submission)
        score = comovement_score(answer, submission)
        
        print(f"\n📊 평가 결과:")
        print(f"  F1 Score: {f1:.6f}")
        print(f"  NMAE: {nmae:.6f}")
        print(f"  Final Score: {score:.6f}")
        print(f"\n  (Score = 0.6 × F1 + 0.4 × (1 - NMAE))")
        print(f"  = 0.6 × {f1:.6f} + 0.4 × {1-nmae:.6f}")
        print(f"  = {score:.6f}")
        
        # 상세 정보
        print(f"\n📈 상세 정보:")
        print(f"  정답 쌍 수: {len(answer)}")
        print(f"  예측 쌍 수: {len(submission)}")
        
        # TP, FP, FN 계산
        ans_pairs = set(zip(answer["leading_item_id"], answer["following_item_id"]))
        sub_pairs = set(zip(submission["leading_item_id"], submission["following_item_id"]))
        tp = len(ans_pairs & sub_pairs)
        fp = len(sub_pairs - ans_pairs)
        fn = len(ans_pairs - sub_pairs)
        
        print(f"  TP (True Positive): {tp}")
        print(f"  FP (False Positive): {fp}")
        print(f"  FN (False Negative): {fn}")
        
    except Exception as e:
        print(f"점수 계산 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    
    # 10. 결과 저장
    print("\n[10단계] 결과 저장 중...")
    output_dir = '../data/validation'
    os.makedirs(output_dir, exist_ok=True)
    
    submission_path = os.path.join(output_dir, 'submission.csv')
    answer_path = os.path.join(output_dir, 'answer.csv')
    
    submission.to_csv(submission_path, index=False)
    answer.to_csv(answer_path, index=False)
    
    print(f"예측 파일 저장: {submission_path}")
    print(f"정답 파일 저장: {answer_path}")
    
    print("\n✅ 검증 완료!")


if __name__ == "__main__":
    main()

