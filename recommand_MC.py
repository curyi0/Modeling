import pandas as pd  # 데이터 프레임 처리용
import numpy as np  # 수치 연산용
import pickle  # 데이터 파일 로드용

import implicit
from implicit.cpu.bpr import BayesianPersonalizedRanking
from implicit.evaluation import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD, NMF  # 행렬분해용
# from db_conn.postgres_db import conn_postgres_db  # 데이터베이스 연결용
from scipy.sparse import lil_matrix
from tqdm import tqdm
# from implicit.bpr import BaysianPersonalizedRanking
from threadpoolctl import threadpool_limits
import warnings

warnings.filterwarnings('ignore')

def extract_highrate_data(minimum_rating= 4.0):

    with open("data/ratings.pkl", "rb") as f:
        df = pickle.load(f)
    df.columns = ["user_id", "movie_id", "rating", "time"]

    users= df['user_id'].value_counts().index[:1000] # groupby, 영화시청 빈도수 상위 1000 사람들 /빠른 결과확인을 위해 줄여서 테스트
    movies=df['movie_id'].value_counts().index[:]
    # print(movies)
    data=df[(df['user_id'].isin(users)) & (df['movie_id'].isin(movies)) & (df['rating'] >= minimum_rating)]
    # print(data)
    return data

def svd_predict_model(users, degree):
    """
    SVD(특이값 분해)를 사용한 협업 필터링 추천 시스템

    Args:
      users: 사용자-영화-평점 데이터프레임
      degree: SVD에서 사용할 차원 수 (잠재 요인 개수)

    Returns:
      예측된 평점이 담긴 데이터프레임 (user_id, movie_id, predicted_rating)
    """

    # ================================
    # 1단계: 피벗 테이블 생성
    # ================================
    # 사용자를 행(index), 영화를 열(columns)로 하는 평점 매트릭스 생성
    # fill_value=None으로 설정하여 평점이 없는 경우 NaN으로 처리
    pivot_rating = users.pivot_table(
        index="user_id", columns="movie_id", values="rating", fill_value=None)

    # ================================
    # 2단계: 빈 평점(Nan) 데이터 처리
    # ================================
    # 각 영화별 평균 평점 계산 (NaN 제외)
    random_mean = pivot_rating.mean(axis=0)

    # 빈 평점을 해당 영화의 평균 평점으로 채우기
    pivot_rating.fillna(random_mean, inplace=True)

    # ================================
    # 3단계: SVD 행렬 분해 준비
    # ================================
    # DataFrame을 numpy 배열로 변환 (SVD 알고리즘 입력용)
    matrix = pivot_rating.values

    # ================================
    # 4단계: SVD 행렬 분해 실행
    # ================================
    # TruncatedSVD: 큰 행렬을 효율적으로 특이값 분해하는 알고리즘
    # n_components=degree: 잠재 요인의 개수 (차원 축소)
    # random_state=42: 재현 가능한 결과를 위한 시드값
    svd = TruncatedSVD(n_components=degree, random_state=42)

    # fit_transform: 사용자 잠재 요인 행렬 생성
    # 각 사용자를 'degree'개의 잠재 요인으로 표현
    user_latent_matrix = svd.fit_transform(matrix)

    # components_: 영화 잠재 요인 행렬
    # 각 영화를 'degree'개의 잠재 요인으로 표현
    item_latent_matrix = svd.components_

    # ================================
    # 5단계: 예측 평점 계산
    # ================================
    # 행렬 곱셈으로 모든 사용자-영화 조합의 예측 평점 계산
    # user_latent_matrix @ item_latent_matrix = 예측 평점 행렬
    predicted_ratings = user_latent_matrix @ item_latent_matrix

    # ================================
    # 6단계: 결과를 DataFrame으로 변환
    # ================================
    # 원본 데이터의 고유한 사용자 ID와 영화 ID 추출
    index = users["user_id"].unique()  # 행 인덱스: 사용자 ID
    columns = users["movie_id"].unique()  # 열 인덱스: 영화 ID

    # 예측 평점을 DataFrame으로 변환 (피벗 테이블 형태)
    predicted_rating_df = pd.DataFrame(
        predicted_ratings, index=index, columns=columns)

    # ================================
    # 7단계: 피벗 해제 (Unpivot)
    # ================================
    # stack(): 피벗 테이블을 긴 형태(long format)로 변환
    # reset_index(): MultiIndex를 일반 컬럼으로 변환
    unpivot_predicted_rating_df = predicted_rating_df.stack().reset_index()

    # 컬럼명을 명확하게 설정
    unpivot_predicted_rating_df.columns = ["user_id", "movie_id", "predicted_rating"]

    return unpivot_predicted_rating_df


def nmf_predict_model(users, degree):
    """
    NMF(비음수 행렬 분해)를 사용한 협업 필터링 추천 시스템

    Args:
      users: 사용자-영화-평점 데이터프레임
      degree: NMF에서 사용할 차원 수 (잠재 요인 개수)

    Returns:
      예측된 평점이 담긴 데이터프레임 (user_id, movie_id, predicted_rating)
    """

    # ================================
    # 1단계: 피벗 테이블 생성
    # ================================
    # 사용자를 행(index), 영화를 열(columns)로 하는 평점 매트릭스 생성
    # fill_value=None으로 설정하여 평점이 없는 경우 NaN으로 처리
    pivot_rating = users.pivot_table(
        index="user_id", columns="movie_id", values="rating", fill_value=None)

    # ================================
    # 2단계: 빈 평점(NaN) 데이터 처리
    # ================================
    # 각 영화별 평균 평점 계산 (NaN 제외)
    random_mean = pivot_rating.mean(axis=0)

    # 빈 평점을 해당 영화의 평균 평점으로 채우기
    pivot_rating.fillna(random_mean, inplace=True)

    # ================================
    # 3단계: NMF 행렬 분해 준비
    # ================================
    # DataFrame을 numpy 배열로 변환 (NMF 알고리즘 입력용)
    matrix = pivot_rating.values

    # NMF는 비음수(non-negative) 값만 허용하므로 음수가 있으면 0으로 처리
    # 평점 데이터는 일반적으로 양수이므로 대부분 문제없음
    matrix = np.maximum(matrix, 0)

    # ================================
    # 4단계: NMF 행렬 분해 실행
    # ================================
    # NMF: 행렬을 두 개의 비음수 행렬의 곱으로 분해하는 알고리즘
    # n_components=degree: 잠재 요인의 개수 (차원 축소)
    # init='random': 랜덤 초기화 방법
    # max_iter=500: 최대 반복 횟수
    # tol=1e-5: 수렴 허용 오차
    # random_state=42: 재현 가능한 결과를 위한 시드값
    nmf = NMF(n_components=degree, random_state=42, init='random', max_iter=500, tol=1e-5)

    # fit_transform: 사용자 잠재 요인 행렬 생성
    # 각 사용자를 'degree'개의 잠재 요인으로 표현
    P = nmf.fit_transform(matrix)

    # components_: 영화 잠재 요인 행렬
    # 각 영화를 'degree'개의 잠재 요인으로 표현
    Q = nmf.components_

    # ================================
    # 5단계: 예측 평점 계산
    # ================================
    # 행렬 곱셈으로 모든 사용자-영화 조합의 예측 평점 계산
    predicted_ratings = P @ Q

    # ================================
    # 6단계: 결과를 DataFrame으로 변환
    # ================================
    # 원본 데이터의 고유한 사용자 ID와 영화 ID 추출
    index = users["user_id"].unique()  # 행 인덱스: 사용자 ID
    columns = users["movie_id"].unique()  # 열 인덱스: 영화 ID

    # 예측 평점을 DataFrame으로 변환 (피벗 테이블 형태)
    predicted_rating_df = pd.DataFrame(
        predicted_ratings, index=index, columns=columns)

    # ================================
    # 7단계: 피벗 해제 (Unpivot)
    # ================================
    # stack(): 피벗 테이블을 긴 형태(long format)로 변환
    # reset_index(): MultiIndex를 일반 컬럼으로 변환
    unpivot_predicted_rating_df = predicted_rating_df.stack().reset_index()

    # 컬럼명을 명확하게 설정
    unpivot_predicted_rating_df.columns = ["user_id", "movie_id", "predicted_rating"]

    return unpivot_predicted_rating_df


def imf_predict_model(users, factors=10, minimum_num_ratings=4, epochs=50):
    """
    IMF(Implicit Matrix Factorization)를 사용한 협업 필터링 추천 시스템

    Args:
      users: 사용자-영화-평점 데이터프레임
      factors: ALS에서 사용할 잠재 요인 수
      minimum_num_ratings: 최소 평점 개수 (이보다 적은 상호작용을 가진 사용자/영화 제외)
      epochs: 학습 반복 횟수

    Returns:
      각 사용자별 상위 N개 추천 영화 리스트
    """

    # ================================
    # 1단계: 데이터 필터링
    # ================================
    # 최소 평점 개수 이상의 상호작용을 가진 사용자만 선택
    user_counts = users["user_id"].value_counts()
    valid_users = user_counts[user_counts >= minimum_num_ratings].index

    # 최소 평점 개수 이상의 상호작용을 가진 영화만 선택
    movie_counts = users["movie_id"].value_counts()
    valid_movies = movie_counts[movie_counts >= minimum_num_ratings].index

    # 필터링된 데이터만 사용
    filtered_users = users[
        (users["user_id"].isin(valid_users)) & (users["movie_id"].isin(valid_movies))]

    # ================================
    # 2단계: 인덱스 매핑 생성
    # ================================
    # 필터링된 데이터를 기반으로 인덱스 매핑 생성
    num_users = filtered_users["user_id"].nunique()
    num_movies = filtered_users["movie_id"].nunique()

    user_id2index = {
        user_id: i for i, user_id in enumerate(filtered_users["user_id"].unique())}
    movie_id2index = {
        movie_id: i for i, movie_id in enumerate(filtered_users["movie_id"].unique())}

    # ================================
    # 3단계: 희소 행렬 생성
    # ================================
    # 사용자-영화 상호작용 행렬 생성
    matrix = lil_matrix((num_users, num_movies))

    # 모든 평점을 1.0으로 변환 (상호작용 여부만 고려)
    for _, row in tqdm(filtered_users.iterrows(), total=len(filtered_users)):
        user_idx = user_id2index[row["user_id"]]
        movie_idx = movie_id2index[row["movie_id"]]
        matrix[user_idx, movie_idx] = 1.0

    # ================================
    # 4단계: CSR 형태로 변환
    # ================================
    # 희소 행렬을 CSR(Compressed Sparse Row) 형태로 변환 (연산 효율성)
    matrix_csr = matrix.tocsr()

    # ================================
    # 5단계: ALS 모델 학습
    # ================================
    # AlternatingLeastSquares: implicit feedback을 위한 행렬분해 알고리즘
    # factors: 잠재 요인의 개수 (차원 축소)
    # iterations: 학습 반복 횟수
    # calculate_training_loss: 학습 손실 계산 여부
    # random_state: 재현 가능한 결과를 위한 시드값
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=epochs,
        calculate_training_loss=True,
        random_state=42
    )

    # ================================
    # 6단계: 모델 학습 실행
    # ================================
    # threadpool_limits: BLAS 스레드 수 제한 (메모리 안정성)
    with threadpool_limits(limits=4, user_api="blas"):
        model.fit(matrix_csr)

    # ================================
    # 7단계: 추천 결과 생성
    # ================================
    # 각 사용자별 상위 N개 영화 추천
    predicted_model = model.recommend_all(matrix_csr, N=10)

    # ================================
    # 8단계: DataFrame 형태로 변환
    # ================================
    # 추천 결과를 저장할 리스트
    recommendations = []

    # 인덱스를 원본 ID로 변환하기 위한 역매핑 생성
    index2user_id = {v: k for k, v in user_id2index.items()}
    index2movie_id = {v: k for k, v in movie_id2index.items()}

    # recommend_all 결과를 올바른 형태로 변환
    for user_idx in range(len(predicted_model)):
        original_user_id = index2user_id[user_idx]
        user_recommendations = predicted_model[user_idx]

        # 각 사용자의 추천 영화들을 처리
        for rank, movie_idx in enumerate(user_recommendations):
            original_movie_id = index2movie_id[movie_idx]

            # 추천 점수는 순위 기반으로 계산 후 0~5점 범위로 스케일링
            # 1위가 5점, 마지막 순위가 0점에 가깝게 설정
            normalized_score = 1.0 - (rank / len(user_recommendations))  # 0~1 범위
            predicted_score = normalized_score * 5.0  # 0~5점 범위로 변환

            recommendations.append({
                'user_id': original_user_id,
                'movie_id': original_movie_id,
                'predicted_rating': float(predicted_score)
            })

    # DataFrame으로 변환
    predicted_df = pd.DataFrame(recommendations)

    return predicted_df

def bpr_predict_model(users, factors=10, minimum_num_ratings=4, epochs=50):
    """
        BPR 모델 기반 추천 예측 함수

        Parameters:
            users (pd.DataFrame): userID, movieID, rating 컬럼 포함된 데이터프레임
            factors (int): 잠재 요인 수 (default: 10)
            minimum_num_ratings (int): 최소 시청한 영화 수를 만족하는 유저만 학습에 사용
            epochs (int): 학습 반복 횟수

        Returns:
            pd.DataFrame: userID, movieID, predicted_score로 구성된 추천 결과
        """

    # 1. 데이터 전처리: 평점 → 암시적 피드백
    df = users.copy()
    df = df[df["rating"] > 0]  # 0은 무시
    df["interaction"] = (df["rating"] >= 3.5).astype(int)

    # 2. 최소 시청 수 기준 필터링
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= minimum_num_ratings].index
    df = df[df["user_id"].isin(valid_users)]

    # 3. ID 인코딩
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df["user"] = user_encoder.fit_transform(df["user_id"])
    df["item"] = item_encoder.fit_transform(df["movie_id"])

    # 4. 희소 행렬 생성
    interactions = csr_matrix((df["interaction"], (df["user"], df["item"])))

    # 5. BPR 모델 훈련
    model = BayesianPersonalizedRanking(factors=factors, iterations=epochs)
    model.fit(interactions)

    # 6. 전체 예측 생성
    all_user_ids = df["user_id"].unique()
    all_item_ids = df["movie_id"].unique()

    predictions = []

    for user_orig in all_user_ids:
        user_idx = user_encoder.transform([user_orig])[0]
        recommended = model.recommend(user_idx, interactions[user_idx], N=10, filter_already_liked_items=True)
        for item_idx, score in recommended:
            movie_id = item_encoder.inverse_transform([item_idx])[0]
            predictions.append((user_orig, movie_id, score))


if __name__ == "__main__":
    # ================================
    # 1단계: 데이터 로드
    # ================================
    # pickle 파일에서 평점 데이터 로드
    with open("data/ratings.pkl", "rb") as f:
        df = pickle.load(f)
    # ================================
    # 2단계: 데이터 전처리
    # ================================
    # 컬럼명을 명확하게 설정
    df.columns = ["user_id", "movie_id", "rating", "time"]
    # print(df.head(5))
    # exit()
    # ================================
    # 3단계: 사용자 필터링
    # ================================
    # 사용자 ID가 1~1001 범위인 데이터만 선택
    # 이는 데이터 크기를 줄여 처리 속도를 높이기 위함
    # users = df[(df["user_id"] >= 1) & (df["user_id"] <= 1001)]

    # ================================
    # 4단계: SVD 추천 모델 실행
    # ================================
    # degree=10: 10개의 잠재 요인으로 차원 축소
    # 결과: 모든 사용자-영화 조합의 예측 평점
    # users_df = svd_predict_model(users, 10)
    # users_df = nmf_predict_model(users, 10)
    users=extract_highrate_data()
    users_df = imf_predict_model(users)
    # users_df = bpr_predict_model(users)

    # 전체 DataFrame 출력 설정
    # pd.set_option('display.max_rows', None)  # 모든 행 출력
    # pd.set_option('display.max_columns', None)  # 모든 열 출력
    # pd.set_option('display.width', None)  # 너비 제한 없음
    # pd.set_option('display.max_colwidth', None)  # 열 너비 제한 없음

    print(users_df)

    # ================================
    # 5단계: 데이터베이스에 저장
    # ================================
    # conn_postgres_db(users_df, "oreo", "1111", "mydb", "nmf_model")