import networkx as nx
import matplotlib.pyplot as plt
from qiskit.transpiler import CouplingMap

# 1. 큐비트 노드 및 위치(좌표) 정의
# 데이터 큐비트 (0~8): 반듯한 3x3 격자
data_qubits = [0, 1, 2, 3, 4, 5, 6, 7, 8]
data_pos = {
    0: (0, 2),  1: (2, 2),  2: (4, 2),
    3: (0, 0),  4: (2, 0),  5: (4, 0),
    6: (0, -2), 7: (2, -2), 8: (4, -2)
}

# 측정 큐비트 (9~16): X-타입(9~12), Z-타입(13~16)
# 내부(Weight-4) 및 테두리(Weight-2) 위치 할당
x_measure_qubits = [9, 10, 11, 12]
x_pos = {
    9: (1, 1),    # 내부 X1 (0, 1, 3, 4 연결)
    10: (3, -1),  # 내부 X2 (4, 5, 7, 8 연결)
    11: (-1, -1), # 좌측 테두리 X (3, 6 연결)
    12: (5, 1)    # 우측 테두리 X (2, 5 연결)
}

z_measure_qubits = [13, 14, 15, 16]
z_pos = {
    13: (3, 1),   # 내부 Z1 (1, 2, 4, 5 연결)
    14: (1, -1),  # 내부 Z2 (3, 4, 6, 7 연결)
    15: (1, 3),   # 상단 테두리 Z (0, 1 연결)
    16: (3, -3)   # 하단 테두리 Z (7, 8 연결)
}

# 전체 위치 정보 병합
pos = {**data_pos, **x_pos, **z_pos}

# 2. 큐비트 간의 물리적 연결(Edge) 정의
edges = [
    # X-타입 측정 큐비트의 연결
    (9, 0), (9, 1), (9, 3), (9, 4),
    (10, 4), (10, 5), (10, 7), (10, 8),
    (11, 3), (11, 6),
    (12, 2), (12, 5),
    
    # Z-타입 측정 큐비트의 연결
    (13, 1), (13, 2), (13, 4), (13, 5),
    (14, 3), (14, 4), (14, 6), (14, 7),
    (15, 0), (15, 1),
    (16, 7), (16, 8)
]

# 3. 최신 Qiskit CouplingMap 객체 생성
# 양자 회로 컴파일 및 트랜스파일 시 하드웨어 제약 조건으로 사용됩니다.
coupling_map = CouplingMap(edges)
coupling_map.make_symmetric() # CNOT 게이트를 위한 양방향 연결 허용

# 4. NetworkX를 이용한 아키텍처 시각화
G = nx.Graph()
G.add_nodes_from(data_qubits)
G.add_nodes_from(x_measure_qubits)
G.add_nodes_from(z_measure_qubits)
G.add_edges_from(edges)

plt.figure(figsize=(9, 9))

# 각 큐비트 타입별로 색상을 다르게 지정하여 노드 그리기
nx.draw_networkx_nodes(G, pos, nodelist=data_qubits, node_color='white', edgecolors='black', node_size=1000, label='Data Qubit')
nx.draw_networkx_nodes(G, pos, nodelist=x_measure_qubits, node_color='gold', edgecolors='black', node_size=1000, label='X-Measure Qubit')
nx.draw_networkx_nodes(G, pos, nodelist=z_measure_qubits, node_color='mediumseagreen', edgecolors='black', node_size=1000, label='Z-Measure Qubit')

# 엣지(연결선) 및 인덱스 번호 그리기
nx.draw_networkx_edges(G, pos, width=2.5, alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif", font_weight='bold')

# Pyplot 설정 (영어 라벨 사용 규칙 적용)
plt.title("d=3 Rotated Surface Code Architecture", fontsize=18, fontweight='bold', pad=20)
plt.legend(scatterpoints=1, loc='upper right', fontsize=11)
plt.axis('off')
plt.tight_layout()
plt.show()

# Qiskit CouplingMap 결과 출력
print(f"Total Qubits in CouplingMap: {coupling_map.size()}")
print("Coupling Map representation successfully created.")