# 1. 예선 리더보드
# 평가 산식 : Score = 0.6 × F1 + 0.4 × (1 − NMAE) [코드]
# 			1) F1 = (2 × Precision × Recall) ÷ (Precision + Recall)

# Precision = TP ÷ (TP + FP)
# Recall = TP ÷ (TP + FN)
# 					여기서

# TP(True Positive): 정답과 예측 모두에 포함된 공행성쌍
# FP(False Positive): 예측에는 있으나 정답에는 없는 쌍
# FN(False Negative): 정답에는 있으나 예측에 없는 쌍

# 			2) NMAE = (1 / |U|) × Σ[min(1, |y_true - y_pred| ÷ (|y_true| + ε))]

# U = 정답 쌍(G)과 예측 쌍(P)의 합집합
# y_true: 정답의 다음달 무역량 (정수 변환)
# y_pred: 예측 무역량 (정수 반올림)
# FN 또는 FP에 해당하는 경우 오차 1.0(100%, 최하점)로 처리
# 오차가 100%를 초과하는 경우에도 1.0(100%, 최하점)로 처리


# Public score : 전체 테스트 데이터 100%
# Private score : 예선 종료 시점의 Public score


# 2. 평가 방식
# 예선 평가 : 예선 리더보드 Private 상위 20팀 선발
# 본선 평가 : 예선 Private 리더보드 점수 50% + 본선 Private 리더보드 점수 50%
# 모델 성능 항목 환산식 : 50 × ((팀의 Private 리더보드 점수) / (최고 점수)) ^ N
# 					※ '최고 점수'는 최종 평가 대상자 중 Private 리더보드 순위가 가장 높은 팀의 점수를 기준으로 하며, N은 1~5 사이의 비공개 조정 계수로 설정

# 					※ 본선 리더보드는 본선 진출자 대상으로 별도의 페이지에서 제공 예정
import numpy as np
import pandas as pd


def _validate_input(answer_df, submission_df):
    # ① 컬럼 개수·이름 일치 여부
    if len(answer_df.columns) != len(submission_df.columns) or not all(answer_df.columns == submission_df.columns):
        raise ValueError("The columns of the answer and submission dataframes do not match.")


    # ② 필수 컬럼에 NaN 존재 여부
    if submission_df.isnull().values.any():
        raise ValueError("The submission dataframe contains missing values.")


    # ③ pair 중복 여부
    pairs = list(zip(submission_df["leading_item_id"], submission_df["following_item_id"]))
    if len(pairs) != len(set(pairs)):
        raise ValueError("The submission dataframe contains duplicate (leading_item_id, following_item_id) pairs.")
        
def comovement_f1(answer_df, submission_df):
    """공행성쌍 F1 계산"""
    ans = answer_df[["leading_item_id", "following_item_id"]].copy()
    sub = submission_df[["leading_item_id", "following_item_id"]].copy()


    ans["pair"] = list(zip(ans["leading_item_id"], ans["following_item_id"]))
    sub["pair"] = list(zip(sub["leading_item_id"], sub["following_item_id"]))


    G = set(ans["pair"])
    P = set(sub["pair"])


    tp = len(G & P)
    fp = len(P - G)
    fn = len(G - P)


    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


    return f1


def comovement_nmae(answer_df, submission_df, eps=1e-6):
    """
    전체 U = G ∪ P에 대한 clipped NMAE 계산
    """
    ans = answer_df[["leading_item_id", "following_item_id", "value"]].copy()
    sub = submission_df[["leading_item_id", "following_item_id", "value"]].copy()


    ans["pair"] = list(zip(ans["leading_item_id"], ans["following_item_id"]))
    sub["pair"] = list(zip(sub["leading_item_id"], sub["following_item_id"]))


    G = set(ans["pair"])
    P = set(sub["pair"])
    U = G | P


    ans_val = dict(zip(ans["pair"], ans["value"]))
    sub_val = dict(zip(sub["pair"], sub["value"]))


    errors = []
    for pair in U:
        if pair in G and pair in P:
            # 정수 변환(반올림)
            y_true = int(round(float(ans_val[pair])))
            y_pred = int(round(float(sub_val[pair])))
            rel_err = abs(y_true - y_pred) / (abs(y_true) + eps)
            rel_err = min(rel_err, 1.0) # 오차 100% 이상은 100%로 간주
        else:
            rel_err = 1.0  # FN, FP는 오차 100%
        errors.append(rel_err)


    return np.mean(errors) if errors else 1.0


def comovement_score(answer_df, submission_df):
    _validate_input(answer_df, submission_df)
    S1 = comovement_f1(answer_df, submission_df)
    nmae_full = comovement_nmae(answer_df, submission_df, 1e-6)
    S2 = 1 - nmae_full
    score = 0.6 * S1 + 0.4 * S2
    return score
