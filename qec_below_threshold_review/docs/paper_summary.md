# Quantum Error Correction Below the Surface Code Threshold — 논문 요약

## 1. 논문 정보

- **제목:** *Quantum error correction below the surface code threshold*
- **저자:** Google Quantum AI and Collaborators
- **학술지:** Nature, Vol. 638
- **온라인 공개:** 2024년 12월 9일
- **인쇄판:** 2025년 2월 27일
- **대상 하드웨어:** Google의 초전도 양자 프로세서 **Willow**
- **DOI:** `10.1038/s41586-024-08449-y`
- **현재 보관된 자료:** Nature 본문 8쪽, Supplementary Information 27쪽, 2026년 저자 정정 1쪽
- **저자 정정:** 2026년 4월 28일 공개. Fig. 3a의 x축을 `Surface code distance`에서 `Repetition code distance`로, 범례를 `d=3 (ref. 17)`/`d=5`에서 `Ref. 17`/`This work`로 바로잡았으며 핵심 수치는 변경하지 않음

---

## 2. 한 문장 요약

이 논문은 실제 초전도 양자 프로세서에서 surface code의 코드 거리 $d$를 증가시킬수록 논리 오류율이 감소하는 **below-threshold 동작**을 실험적으로 보이고, 논리 메모리의 수명이 가장 좋은 개별 물리 큐비트보다 길어지는 **beyond-breakeven memory**와 **실시간 복호**까지 구현한 연구이다.

---

## 3. 연구 배경과 핵심 문제

양자 오류 정정에서는 여러 물리 큐비트를 하나의 논리 큐비트로 묶어 사용한다. 이상적으로는 물리 연산의 오류율이 임계값인 threshold보다 낮을 때, 코드 거리를 증가시킬수록 논리 오류율이 지수적으로 감소해야 한다.

논문에서는 surface code의 논리 오류율을 다음과 같이 근사한다.

$$
\varepsilon_d
\propto
\left(
\frac{p}{p_{\mathrm{thr}}}
\right)^{(d+1)/2}
$$

여기서

- $p$: 물리 오류율
- $p_{\mathrm{thr}}$: threshold error rate
- $d$: code distance
- $\varepsilon_d$: QEC cycle당 논리 오류율

이다.

코드 거리를 2 증가시켰을 때의 오류 억제 계수는

$$
\Lambda =
\frac{\varepsilon_d}{\varepsilon_{d+2}}
$$

로 정의한다.

- $\Lambda>1$: 코드 거리를 증가시키면 논리 오류율이 감소
- $\Lambda=2$: 코드 거리를 2 증가시킬 때 논리 오류율이 절반으로 감소
- $\Lambda<1$: 코드 크기를 키워도 성능이 악화

따라서 이 논문의 핵심 목표는 실제 프로세서에서 $\Lambda>1$, 특히 $\Lambda\approx2$ 이상의 below-threshold 성능을 확인하는 것이다.

---

## 4. 실험 시스템

논문에서는 두 개의 Willow 프로세서를 사용했다.

| 프로세서 | 구현 코드 | 주요 목적 |
|---|---:|---|
| 105-qubit Willow | distance-7 surface code | 고정밀 below-threshold 성능 측정 |
| 72-qubit Willow | distance-5 surface code | 실시간 복호, leakage 제거, 장시간 안정성 실험 |

### 4.1 Distance-7 surface code 구성

Distance-7 논리 메모리는 다음과 같이 구성된다.

- 데이터 큐비트: 49개
- 측정 큐비트: 48개
- 추가 leakage removal 큐비트: 4개
- 총 사용 큐비트: 101개

데이터 큐비트는 $7\times7$ 격자를 이루며, 측정 큐비트가 stabilizer parity 정보를 반복적으로 추출한다.

### 4.2 사용한 코드

논문 본문은 사용 코드를 **ZXXZ surface code**라고 표기한다. Supplementary Fig. S15도 모든 stabilizer가 ZXXZ를 측정하며, 그 안에서 사용하는 “X stabilizer”와 “Z stabilizer”라는 이름은 CZ gate ordering과 logical basis를 구분하기 위한 convention이라고 설명한다. 참고문헌 31의 제목은 *The XZZX surface code*이지만, 이 리뷰에서는 실험 회로에 대한 원문 표기인 ZXXZ를 사용한다.

논리 상태는 $X_L$ 또는 $Z_L$ basis의 고유상태로 초기화되며, 반복적인 syndrome extraction 후 decoder가 최종 논리 측정값을 재해석한다.

오류를 반드시 물리적으로 즉시 보정하는 것은 아니다. Decoder가 오류 정보를 추적한 뒤 최종 논리 측정 결과를 수정하는 **Pauli-frame 방식**을 사용할 수 있다.

---

## 5. 주요 실험 결과

## 5.1 Below-threshold scaling

논문은 각 논리 basis와 코드에 대해 최대 250 QEC cycle의 logical error probability를 fitting하여 cycle당 논리 오류율을 구했다. Distance-3은 9개, distance-5는 4개 하위 격자의 결과를 평균했고, $X_L$과 $Z_L$ basis를 모두 평균했다. 최종 $Lambda$는 $ln \varepsilon_d$를 $d$에 대해 선형 회귀하여 계산했다.

Neural-network decoder를 적용했을 때 논문은 다음 결과를 얻었다.

$$
\Lambda = 2.14 \pm 0.02
$$

Distance-7 코드의 QEC cycle당 논리 오류율은

$$
\varepsilon_7
=
(1.43\pm0.03)\times10^{-3}
$$

즉,

$$
\varepsilon_7
=
0.143\%\pm0.003\%
$$

이다.

Ensembled matching-synthesis decoder를 사용했을 때는

$$
\Lambda=2.04\pm0.02
$$

$$
\varepsilon_7
=
(1.71\pm0.03)\times10^{-3}
$$

를 얻었다.

두 고정밀 decoder 모두 $\Lambda>2$를 기록했으므로, $d=3\rightarrow5\rightarrow7$로 코드 거리를 증가시킬 때 논리 오류율이 지속적으로 절반 이하로 감소했다.

이는 surface code가 실제 하드웨어에서 threshold 아래에서 동작한다는 핵심 증거이다.

이때 평균 bulk detection probability는

$$
p_{\mathrm{det}}=(7.7\%,\,8.5\%,\,8.7\%)
$$

로, 각각 $d=(3,5,7)$에 대응한다. 코드 거리가 커질수록 detection probability가 조금 증가했음에도 논리 오류율은 감소했다. 논문은 이 증가를 finite-size effect와 parasitic coupling의 영향으로 해석한다.

측정된 component error, leakage, stray interaction을 사용한 simulation은 distance 11까지 수행되었고,

$$
\Lambda_{\mathrm{sim}}=2.25\pm0.02
$$

를 얻었다. 실험과 simulation의 일치는 below-threshold 해석을 지지하지만, simulation은 실제 장치의 모든 오류를 설명하지는 못한다.

### 5.1.1 AlphaQubit 명칭에 대한 구분

Willow 논문은 고정밀 decoder를 본문과 SI에서 **neural-network decoder**라고 부르며, Bausch et al.의 별도 Nature 논문 *Learning high-accuracy error decoding for quantum processors*를 인용한다. 이 별도 논문이 recurrent-transformer architecture를 **AlphaQubit**이라고 명명한다.

AlphaQubit 논문은 Sycamore의 distance-3/5 experimental data와 최대 distance-11 simulation을 중심으로 한다. Willow SI는 synthetic pretraining, device-specific DEM과 experimental-data fine-tuning, 장시간 decoding용 gated recurrence를 설명하지만 검색 가능한 본문/SI에는 `AlphaQubit`이라는 고유명칭이 직접 나오지 않는다.

따라서 이 리뷰에서는 다음처럼 표현한다.

> **Willow 원문 표기인 neural-network decoder이며, 관련 선행 architecture는 AlphaQubit 논문에 제시되었다.**

AlphaQubit은 논문 리뷰에서 다루되, 본 프로젝트의 PyMatching simulation이 AlphaQubit을 구현하거나 재현했다고 표현하지 않는다.

---

## 5.2 Beyond-breakeven logical memory

Distance-7 논리 큐비트의 수명은

$$
291\pm6\,\mu\mathrm{s}
$$

였다.

같은 논리 큐비트를 구성하는 물리 큐비트의 수명은 다음과 같다.

- 물리 큐비트 수명 중앙값: $85\pm7\,\mu\mathrm{s}$
- 가장 좋은 물리 큐비트 수명: $119\pm13\,\mu\mathrm{s}$

따라서 distance-7 논리 메모리는 가장 좋은 구성 물리 큐비트보다

$$
2.4\pm0.3
$$

배 오래 양자 정보를 보존했다.

이를 **beyond breakeven**이라고 한다.

다만 이 결과는 논리 **메모리**의 breakeven이다. 여러 논리 게이트와 feed-forward를 포함한 완전한 논리 연산의 breakeven을 달성했다는 뜻은 아니다.

---

## 6. 논리 오류 민감도 분석

## 6.1 물리 오류 주입 실험

논문은 데이터 큐비트와 측정 큐비트에 크기를 조절할 수 있는 coherent error를 인위적으로 주입하고, 물리 오류 수준과 논리 오류율 사이의 관계를 측정했다.

Distance $d=3,5,7$의 논리 오류 곡선은 detection probability 약 20% 부근에서 교차했다.

- 교차점 아래에서는 큰 거리의 코드가 더 낮은 논리 오류율을 보임
- 교차점 위에서는 코드 크기를 키워도 성능이 개선되지 않음

이는 threshold 동작의 전형적인 특징이다.

이상적인 독립 오류 모델에서는 논리 오류율이 대략

$$
O\left(p^{(d+1)/2}\right)
$$

로 감소해야 한다. 실제 실험에서는 이상적인 지수의 약 80% 수준이 관측되었으며, 논문은 초과된 correlated error를 주요 원인으로 추정한다.

---

## 6.2 Error budget

논문은 실제 프로세서의 component error와 noise simulation을 이용해 $1/\Lambda$에 대한 error budget을 구성했다.

이 분석은 72-qubit processor와 correlated matching decoder를 기준으로 수행되었다. 같은 decoder를 실험 데이터에 적용하면

$$
\Lambda=1.97\pm0.02
$$

였고, noise model은 $\Lambda$를 약 14% 과대예측했다. 즉 모델이 대부분의 오류를 포착했지만, 장거리 상호작용이나 고에너지 leakage 같은 out-of-model error가 남아 있을 가능성이 있다.

포함된 주요 오류원은 다음과 같다.

- CZ 게이트의 일반적인 오류
- CZ 게이트 중 인접 큐비트와의 stray interaction
- 데이터 큐비트 idle error
- 측정 및 reset 오류
- leakage
- 단일 큐비트 게이트 오류
- 모델에 포함되지 않은 excess error

전체적으로 **CZ 관련 오류와 데이터 큐비트 idle error**가 큰 비중을 차지했다.

Leakage와 stray interaction 같은 correlated error는 추정 error budget의 약 17%를 차지했다. 현재는 지배적인 오류가 아니지만, local error가 더 낮아질수록 상대적으로 더 중요한 제한 요인이 될 수 있다.

---

## 7. Leakage removal

초전도 transmon은 계산 공간 $\{|0\rangle,|1\rangle\}$ 밖의 $|2\rangle$ 이상 상태로 전이될 수 있다. 이를 leakage라고 한다.

논문에서는 각 syndrome extraction 후 **DQLR(Data Qubit Leakage Removal)**을 수행했다.

DQLR은 데이터 큐비트의 leakage excitation을 측정 큐비트 또는 별도의 leakage removal 큐비트로 옮긴 뒤 reset하는 방식이다.

### DQLR의 효과

- Distance-3 코드에서는 성능 변화가 크지 않음
- Distance-5 코드에서는 논리 성능이 크게 개선됨
- 오류 억제 계수 $\Lambda$가 약 35% 증가
- Detection probability는 약 12%만 감소

이 결과는 detection event의 평균 개수만으로 논리 성능을 완전히 평가할 수 없다는 점을 보여준다.

특히 leakage처럼 시간적·공간적 correlation을 만드는 오류에서는 동일한 detection probability라도 논리 오류에 미치는 영향이 더 클 수 있다.

---

## 8. 장시간 안정성

대규모 fault-tolerant 알고리즘은 수 시간 동안 안정적으로 실행되어야 한다.

논문은 72-qubit 프로세서에서 다음 코드를 약 15시간 동안 반복 실행했다.

- Distance-5 코드 1개
- Distance-3 코드 4개

전체 측정은 16회 수행되었고, 네 번의 실험마다 qubit frequency와 readout drift를 보정하기 위한 재보정을 수행했다.

Neural-network decoder 기준 평균 오류 억제 계수는

$$
\Lambda=2.18\pm0.07
$$

이었고, 가장 좋은 경우

$$
\Lambda=2.31\pm0.02
$$

였다.

큐비트 주파수와 readout 신호의 drift가 존재했지만, distance-5 논리 큐비트의 성능은 비교적 안정적으로 유지되었다.

또한 일부 국소적인 물리 큐비트 성능 변동이 distance-3 코드에서는 뚜렷하게 보였지만, distance-5 코드에서는 영향이 완화되었다.

이는 더 큰 코드가 개별 소자의 drift와 fluctuation에 덜 민감할 수 있음을 보여준다.

---

## 9. 고거리 repetition code와 logical error floor

Surface code만으로 극저오류 영역을 충분히 조사하기 어렵기 때문에 논문은 distance-29까지의 repetition code도 실행했다.

### 실험 조건

- 최대 distance: $d=29$
- 한 shot당 QEC cycle: 1,000회
- 총 반복: bit-flip과 phase-flip 각각 $10^7$회
- 총 QEC cycle: $2\times10^{10}$
- 실제 프로세서 실행 시간: 약 5.5시간

Distance $5$에서 $11$ 사이에서는

$$
\Lambda=8.4\pm0.1
$$

의 강한 오류 억제가 관측되었다.

그러나 $d\geq15$에서는 지수적 오류 감소에서 벗어나기 시작했고, 논리 오류율이 약

$$
10^{-10}
$$

수준에서 포화되는 error floor가 나타났다.

Distance-29에서는 실제 logical failure가 관측되지 않았지만, 논문은 가장 강한 burst에서도 decoder가 우연히 올바른 결과를 낸 영향으로 해석한다. 따라서 $10^{-10}$은 distance-29에서 직접 관측한 실패율이라기보다 여러 하위 distance의 scaling과 failure pattern을 함께 본 apparent error floor이다.

별도의 coherent error injection에서는 detection probability를 절반으로 줄였을 때 distance-15의 논리 오류율이 약 250배 감소했다. 이는 기대되는 $O(p^8)$ scaling과 일치한다.

---

## 10. Rare correlated error burst

고거리 repetition code의 실패 패턴에서는 두 종류의 비정상 오류가 관찰되었다.

### 10.1 국소적인 transient event

- 한두 개 detector의 detection probability가 갑자기 3배 이상 증가
- 수십~수백 cycle 후 원래 수준으로 회복
- 가능한 원인으로 transient TLS 또는 coupler excitation이 제시됨

### 10.2 대규모 correlated burst

- 여러 detector에서 detection probability가 동시에 크게 증가
- 약 30개 큐비트 규모의 인접 영역에 공간적으로 국소화
- 약 한 시간에 한 번 발생
- 총 $2\times10^{10}$ cycle 동안 6회 관찰
- 약 $400\,\mu\mathrm{s}$의 시간상수로 감쇠
- 정확한 원인은 아직 밝혀지지 않음

이 rare correlated burst가 고거리 코드의 logical error floor를 만든다.

평균적인 local gate error를 낮추더라도 이러한 희귀 대규모 오류가 남아 있으면, 충분히 큰 코드에서 기대한 지수적 오류 억제가 중단될 수 있다.

---

## 11. 실시간 복호

Fault-tolerant computation에서는 syndrome 데이터를 생성 속도만큼 빠르게 처리해야 한다.

Willow 프로세서의 QEC cycle 시간은

$$
1.1\,\mu\mathrm{s}
$$

이다.

Decoder의 평균 처리 throughput이 syndrome 생성 속도를 따라가지 못하면 backlog가 계속 쌓이고, 논리 연산의 반응 시간이 길어진다.

### 11.1 실시간 decoding pipeline

논문의 실시간 복호 시스템은 다음 순서로 동작한다.

1. 제어 전자장치가 측정 신호를 bit로 분류
2. 저지연 Ethernet으로 decoder workstation에 전달
3. 측정값을 detection event로 변환
4. Shared-memory buffer를 통해 streaming decoder에 전달
5. **Sparse Blossom** 기반 minimum-weight perfect matching 수행
6. 시공간 graph를 여러 block으로 나누어 병렬 처리
7. 각 block의 결과를 fuse하여 global matching 생성

Decoder는 전체 실험 길이에 비례해 계속 커지는 graph를 저장하지 않고, 일정 크기의 graph buffer만 유지한다.

구현은 10-cycle block을 사용한다. 미래 block과의 fuse로 해결할 수 없는 detection-event 구성이 생기면 decoder가 heralded failure를 반환하며, block 크기는 이 failure rate가 logical failure rate보다 무시할 수 있을 정도로 작아지도록 선택되었다.

이 구조 덕분에 최대 $10^6$ cycle까지 backlog 없이 복호할 수 있었다.

---

## 11.2 Decoder latency

Distance-5 surface code에서 최대

$$
10^6
$$

QEC cycle을 실행했다.

평균 decoder latency는

$$
63\pm17\,\mu\mathrm{s}
$$

였다.

논문에서 latency는 마지막 syndrome cycle이 decoder에 도착한 시점부터 최종 correction을 반환할 때까지의 시간으로 정의된다.

따라서 $63\,\mu\mathrm{s}$는 완전한 end-to-end reaction latency가 아니다. 별도로 추정한 데이터 전송 시간은 $10\,\mu\mathrm{s}$ 미만이며, 실제 quantum feedback은 이 실험에서 아직 구현되지 않았다. 또한 decoder latency는 code size와 함께 증가하고, 드물게 subshot latency가 $1\,\mathrm{ms}$를 넘는 spike도 관측되었다.

실험의 총 cycle 수를 증가시켜도 이 latency가 거의 일정하게 유지되었다. 이는 syndrome 데이터가 실시간으로 처리되고 backlog가 누적되지 않았음을 의미한다.

다만 cycle당 syndrome 처리 throughput과 최종 correction latency는 구분해야 한다.

- **Throughput:** 매 $1.1\,\mu\mathrm{s}$마다 생성되는 syndrome을 지속적으로 처리할 수 있는가
- **Latency:** 마지막 syndrome 이후 최종 correction을 반환하는 데 걸리는 시간

---

## 11.3 실시간 decoder와 offline decoder 비교

| Decoder | $\varepsilon_5$ | $\Lambda$ | 특징 |
|---|---:|---:|---|
| Real-time Sparse Blossom | $0.35\%\pm0.01\%$ | $2.0\pm0.1$ | 빠른 streaming decoding |
| Offline neural network | $0.269\%\pm0.008\%$ | $2.18\pm0.09$ | 더 높은 정확도 |
| Ensembled matching synthesis | 논문 내 별도 비교 | 고정밀 offline | 여러 matching 결과 조합 |

실시간 decoder는 offline neural-network decoder보다 정확도가 낮지만, 여전히 $\Lambda>1$이며 below-threshold 성능을 유지했다.

처리 속도 측면에서 실시간 decoder는 cycle당 $1.1\,\mu\mathrm{s}$보다 빠른 지속 throughput이 필요했지만, 당시 offline neural-network decoder는 cycle당 약 $24\,\mu\mathrm{s}$를 사용했다. 이 수치는 batch 처리시간과 end-to-end latency가 서로 다른 지표임도 함께 보여준다.

이는 현재 QEC decoder 연구의 대표적인 trade-off를 보여준다.

$$
\text{Accuracy}
\quad\leftrightarrow\quad
\text{Latency and throughput}
$$

---

## 12. 논문의 핵심 수치 정리

| 항목 | 결과 |
|---|---:|
| 최대 surface code distance | $d=7$ |
| Distance-7 총 사용 큐비트 | 101개 |
| Neural-network decoder 오류 억제 계수 | $\Lambda=2.14\pm0.02$ |
| Distance-7 논리 오류율 | $0.143\%\pm0.003\%$ per cycle |
| Distance-7 논리 수명 | $291\pm6\,\mu\mathrm{s}$ |
| 최상 물리 큐비트 대비 수명 | $2.4\pm0.3$배 |
| QEC cycle 시간 | $1.1\,\mu\mathrm{s}$ |
| 실시간 decoder 평균 latency | $63\pm17\,\mu\mathrm{s}$ |
| 실시간 decoder $\Lambda$ | $2.0\pm0.1$ |
| 최대 연속 실시간 decoding | $10^6$ cycles |
| 최대 repetition code distance | $d=29$ |
| Repetition-code 오류 억제 계수 | $\Lambda=8.4\pm0.1$ |
| 관측된 repetition-code error floor | 약 $10^{-10}$ per cycle |
| 대규모 correlated burst 빈도 | 약 1시간당 1회 |

---

## 13. 남은 한계

### 13.1 현재 논리 오류율은 실용적 알고리즘에 충분하지 않음

Distance-7의 논리 오류율은 약

$$
1.43\times10^{-3}
$$

per cycle이다.

논문의 외삽에 따르면 현재 장치 성능을 그대로 유지하면서 논리 오류율

$$
10^{-6}
$$

을 달성하려면 distance $d=27$이 필요하며, 논리 큐비트 하나에 약 1,457개의 물리 큐비트가 필요하다.

### 13.2 Decoder 확장성

코드 거리가 증가하면 stabilizer 수와 syndrome 데이터량이 대략 $d^2$에 비례해 증가한다.

따라서 다음 문제가 커진다.

- 데이터 전송 대역폭
- Decoder의 계산량
- Memory bandwidth
- 병렬 처리 자원
- 최종 reaction latency

실험에서 총 cycle 수에 대한 backlog는 억제했지만, decoder latency 자체는 code size에 따라 증가했다. 따라서 $10^6$ cycle 처리 성공이 더 큰 code distance에서도 동일한 latency를 보장하지는 않는다.

### 13.3 Correlated error floor

Rare correlated burst는 독립 Pauli error를 가정한 이상적인 threshold scaling을 무너뜨릴 수 있다.

오류의 원인을 규명하고 하드웨어 또는 decoder 수준에서 완화해야 한다.

### 13.4 논리 메모리만 검증

이 논문은 논리 큐비트의 **memory experiment**에 초점을 둔다.

완전한 fault-tolerant computation을 위해서는 추가로 다음이 필요하다.

- 여러 논리 큐비트
- 논리 CNOT
- Lattice surgery
- Logical state injection
- Non-Clifford gate
- Decoder 결과를 이용한 실시간 feed-forward
- Logical compiler와 decoder의 통합
- 장시간 논리 알고리즘 실행

특히 이번 실시간 복호 실험은 correction을 계산하는 단계까지이며, 계산 결과를 quantum processor에 되돌려 보내는 feedback은 구현하지 않았다.

---

## 14. 논문의 의의

이 논문의 가장 중요한 의의는 실제 초전도 프로세서가 surface code threshold 아래의 영역에 진입했다는 점이다.

즉,

$$
d=3\rightarrow5\rightarrow7
$$

로 코드 거리를 증가시킬수록 논리 오류율이 감소했다.

이는 대규모 fault-tolerant quantum computing이 이미 완성되었다는 뜻은 아니다. 정확한 의미는 다음과 같다.

> 현재의 물리 오류 성능을 유지하거나 더 개선하면서 프로세서와 decoder를 실제로 확장할 수 있다면, 논리 오류율을 계속 감소시킬 수 있는 조건이 실험적으로 확인되었다.

이 연구는 surface code가 이론적인 오류 정정 코드에서 실제 확장 가능한 하드웨어·소프트웨어 시스템으로 넘어가는 중요한 단계이다.

---

## 15. QEC decoder 연구 관점에서의 시사점

이 논문은 decoder 연구에 다음과 같은 시사점을 준다.

### 15.1 정확도만으로는 부족함

Offline neural-network decoder가 가장 높은 정확도를 보였지만, fault-tolerant computation에서는 실시간 throughput과 latency도 필수적이다.

### 15.2 실제 장치 noise에 맞춘 decoder가 필요함

논문은 다음 방법을 사용했다.

- 실제 processor data를 이용한 neural-network fine-tuning
- Reinforcement learning을 통한 matching graph weight 최적화
- Correlated error를 반영한 edge reweighting
- 여러 matching decoder를 결합한 ensemble
- Streaming 및 parallel window 방식

이는 균일한 독립 Pauli noise만을 가정한 decoder로는 실제 프로세서 성능을 충분히 끌어내기 어렵다는 점을 보여준다.

### 15.3 Leakage와 correlated noise가 중요함

Detection probability가 비슷하더라도 leakage 제거 여부에 따라 논리 오류율이 크게 달라졌다.

따라서 decoder 입력과 noise model에는 다음 요소가 중요하다.

- 시간 방향의 correlation
- 공간 방향의 correlation
- Leakage 지속 시간
- CZ-induced correlated error
- Calibration drift
- Rare burst event

### 15.4 평가 지표를 분리해야 함

QEC decoder를 평가할 때는 적어도 다음을 따로 측정해야 한다.

- Logical error rate
- Error suppression factor $\Lambda$
- Cycle throughput
- End-to-end latency
- Memory usage
- Decoder failure 또는 heralded failure rate
- Noise-model mismatch에 대한 robustness
- 장시간 drift에 대한 안정성

이때 논문이 보고한 $63\,\mu\mathrm{s}$는 decoder software가 마지막 syndrome을 받은 뒤 correction을 반환하기까지의 latency이므로, 데이터 전송과 실제 feedback을 포함한 end-to-end latency와 구분해야 한다.

---

## 16. 최종 정리

이 논문은 다음 네 가지를 동시에 보여준다.

1. **Below-threshold scaling**

   $$
   \Lambda=2.14\pm0.02
   $$

2. **Beyond-breakeven logical memory**

   Distance-7 논리 큐비트가 최상 물리 큐비트보다 $2.4\pm0.3$배 긴 수명을 기록했다.

3. **Real-time decoding**

   $1.1\,\mu\mathrm{s}$ QEC cycle에서 최대 $10^6$ cycle을 backlog 없이 처리했다.

4. **남은 핵심 병목 확인**

   Leakage, CZ correlation, drift, rare correlated burst, decoder 확장성이 앞으로 해결해야 할 주요 문제로 확인되었다.

결론적으로 이 연구는 surface code 기반 fault-tolerant quantum computing이 원리적으로만 가능한 단계에서 벗어나, 실제 하드웨어에서 규모 확장을 논의할 수 있는 단계로 진입했음을 보여준다.

---

## 17. 비판적 검토와 후속 확인 과제

### 17.1 직접 관측과 외삽의 구분

- 직접 관측된 surface-code scaling 범위는 $d=3,5,7$이다.
- $d=27$에서 $10^{-6}$을 달성한다는 내용은 현재 noise 수준이 유지된다는 가정의 외삽이다.
- repetition-code의 $10^{-10}$ error floor가 surface code에 동일하게 나타난다고 단정할 수는 없다. Surface code는 더 많은 오류 채널과 다른 공간 구조에 민감하다.

### 17.2 재현 가능한 범위

Stim과 PyMatching을 이용한 일반적인 rotated surface-code simulation은 below-threshold 현상을 정성적으로 검증할 수 있지만, Willow의 ZXXZ schedule, calibrated device noise, neural-network decoder 또는 실시간 streaming stack을 재현하는 것은 아니다. 따라서 후속 구현 결과는 논문의 수치 재현이 아니라 단순화된 모델에서의 현상 검증으로 표현해야 한다.

### 17.3 Supplementary Information 확인 결과와 남은 검토

확인된 사항:

- 실험 회로는 ZXXZ stabilizer를 측정하며 X/Z stabilizer 명칭은 gate ordering convention이다.
- 총 다섯 계열의 decoder를 상황에 따라 사용했다: neural network, Libra matching synthesis, Harmony ensemble, correlated matching, real-time decoder.
- Neural-network decoder는 synthetic pretraining 뒤 DEM과 실험 데이터로 두 단계 fine-tuning하며, 장시간 실험을 위해 gated recurrence를 추가했다.
- Real-time decoder는 distance-3/5에서 10-cycle block, distance-29 repetition code에서 90-cycle block을 사용해 heralded failure를 억제했다.
- 보고된 $63\,\mu\mathrm{s}$는 평균 software latency이며 $T_{\mathrm{input}}<10\,\mu\mathrm{s}$, $T_{\mathrm{output}}$과 실제 control feedback은 포함되지 않는다.

추가로 상세 검토할 사항:

- logical lifetime 및 $\varepsilon_d$ fitting의 통계 절차
- leakage 및 stray-interaction noise model의 simulation parameter
- rare correlated burst의 failure 분류와 error-floor 추정 절차

### 17.4 리뷰의 핵심 판단

이 논문의 가장 강한 결과는 실제 장치에서 $d=3\rightarrow5\rightarrow7$로 증가할 때 논리 오류율이 일관되게 감소했다는 점이다. 반면 대규모 fault-tolerant computation의 실현 가능성은 아직 외삽, correlated error 완화, decoder scaling 및 feedback 통합에 달려 있다.
