import stim
import numpy as np
import time

# 1. Surface Code 시뮬레이션 파라미터 설정
d = 3  # Distance (공간적 크기: 방패의 넓이)
r = 3  # Rounds (시간적 반복 횟수: 방패의 두께)
noise_rate = 0.01  # 물리적 에러율 1% 가정

print(f"=== [d={d}, r={r}, p={noise_rate}] Surface Code 데이터 생성 시작 ===")
start_time = time.time()

# 2. Stim을 이용한 양자 회로 생성
# 'rotated_memory_z'는 Z-편향 잡음 연구에 가장 널리 쓰이는 표준 평면 격자 구조입니다.
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=d,
    rounds=r,
    after_clifford_depolarization=noise_rate
)

# 3. 디텍터 샘플러 컴파일 및 시뮬레이션 실행
# separate_observables=True 를 설정하여 머신러닝의 입력(X)과 정답(Y)을 완벽히 분리합니다.
sampler = circuit.compile_detector_sampler()
shots = 10000  # 1만 개의 훈련 데이터(Shots) 

X_train, Y_train = sampler.sample(shots=shots, separate_observables=True)
end_time = time.time()

# 4. 결과 출력 및 텐서 형태 확인
print(f"데이터 생성 완료! (소요 시간: {end_time - start_time:.4f}초)")
print(f"\n[모델 입력 데이터] X_train (Detector Events) 크기: {X_train.shape}")
print(f"[모델 정답 라벨] Y_train (Observable Flips) 크기: {Y_train.shape}")

# 5. 생성된 데이터를 PyTorch 학습용 파일로 저장
# 어제 설정한 .gitignore 덕분에 이 큰 파일들은 GitHub에 올라가지 않고 안전하게 로컬에만 보관됩니다.
np.save("detector_events_X.npy", X_train)
np.save("logical_flips_Y.npy", Y_train)
print("\n파일 저장 완료: 'detector_events_X.npy', 'logical_flips_Y.npy'")