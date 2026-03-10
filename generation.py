import stim
import numpy as np
import time
import os

def generate_custom_noisy_surface_code(d, r, p_z, p_x, p_crosstalk):
    """
    편향 노이즈(Biased noise)와 크로스토크(Crosstalk)가 반영된 
    실제 하드웨어 환경 수준의 커스텀 표면 코드 회로를 생성합니다.
    """
    # 1. 이상적인(Ideal) 표면 코드 회로 생성
    ideal_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=r,
        after_clifford_depolarization=0.0, 
        before_round_data_depolarization=0.0,
        before_measure_flip_probability=0.0,
        after_reset_flip_probability=0.0
    )
    
    custom_circuit = stim.Circuit()
    
    # 2. 명령어 순회 및 커스텀 노이즈 주입
    for instruction in ideal_circuit:
        name = instruction.name
        targets = instruction.targets_copy()
        args = instruction.gate_args_copy()
        
        custom_circuit.append(name, targets, args)
        
        # 단일 큐비트 게이트 이후 편향 노이즈 발생
        if name in ["I", "H", "R"]:
            custom_circuit.append("PAULI_CHANNEL_1", targets, [p_x, p_x, p_z])
            
        # 2-큐비트 연산 이후 크로스토크 에러 발생
        elif name in ["CX", "CZ"]:
            custom_circuit.append("DEPOLARIZE2", targets, p_crosstalk)
            
        # 측정 과정에서의 에러 발생
        elif name == "M":
            custom_circuit.append("X_ERROR", targets, p_x)

    return custom_circuit

def simulate_and_save_npy(d, r, p_z, p_x, p_crosstalk, shots, save_dir="dataset"):
    """
    회로를 시뮬레이션하고 결과를 numpy 배열(.npy)로 저장합니다.
    """
    print(f"=== [d={d}, r={r}] QEC 데이터 생성 시작 ===")
    start_time = time.time()
    
    # 1. 커스텀 회로 생성
    print("1. 커스텀 노이즈 회로 생성 중...")
    circuit = generate_custom_noisy_surface_code(d, r, p_z, p_x, p_crosstalk)
    
    # 2. 디텍터 샘플러 컴파일
    print(f"2. {shots} Shots 시뮬레이션 진행 중...")
    sampler = circuit.compile_detector_sampler()
    
    # separate_observables=True 로 입력(신드롬)과 정답(논리 오류) 분리
    X_train, Y_train = sampler.sample(shots=shots, separate_observables=True)
    
    # 3. 데이터 저장 폴더 확인 및 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 파일명 설정 (파라미터 정보를 포함하여 관리하기 쉽게 지정)
    x_filename = os.path.join(save_dir, f"X_d{d}_r{r}_pz{p_z}_shots{shots}.npy")
    y_filename = os.path.join(save_dir, f"Y_d{d}_r{r}_pz{p_z}_shots{shots}.npy")
    
    # 4. 불리언(bool) 타입 그대로 저장하여 용량 최소화
    # X_train과 Y_train은 이미 Stim에 의해 numpy bool array로 반환됩니다.
    np.save(x_filename, X_train)
    np.save(y_filename, Y_train)
    
    end_time = time.time()
    
    print("\n=== 데이터 생성 및 저장 완료 ===")
    print(f"소요 시간: {end_time - start_time:.4f}초")
    print(f"[입력 데이터] X_train (신드롬) 크기: {X_train.shape}, 타입: {X_train.dtype}")
    print(f"[정답 라벨] Y_train (논리 오류) 크기: {Y_train.shape}, 타입: {Y_train.dtype}")
    print(f"저장 위치:\n - {x_filename}\n - {y_filename}")

if __name__ == "__main__":
    # --- 파라미터 설정 ---
    distance = 5
    rounds = 5
    
    # 강한 Z-편향 노이즈 환경 가정
    prob_z = 0.005           # Z 에러 0.5%
    prob_x = 0.0005          # X 에러 0.05%
    prob_crosstalk = 0.001   # 크로스토크 0.1%
    
    num_shots = 100000       # 10만 개 데이터 생성
    
    # --- 실행 ---
    simulate_and_save_npy(
        d=distance, 
        r=rounds, 
        p_z=prob_z, 
        p_x=prob_x, 
        p_crosstalk=prob_crosstalk, 
        shots=num_shots
    )