Dataset Info.

train.csv [파일] :
item_id : 무역품 식별 ID
year : 년
month : 월
seq : 동일 년-월 내 일련번호
type : 유형 구분 코드
hs4 : HS4 코드
weight : 중량
quantity : 수량
value : 무역량 (정수형)


sample_submission.csv [파일] - 제출 양식
leading_item_id : 선행 무역품 식별 ID
following_item_id : 후행 무역품 식별 ID
value : 2025년 8월의 후행 무역품에 예측된 총 무역량 (정수형)




※ 제공드리는 데이터를 엑셀로 열람하는 경우, 데이터가 비정상적으로 보이는 현상이 발생할 수 있으니 반드시 Pandas패키지와 같은 데이터툴을 이용하여 열람부탁드립니다.