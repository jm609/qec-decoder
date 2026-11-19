# Willow QEC 논문의 주요 수식과 도출 과정

> 목적: 논문과 Supplementary Information에 등장하는 핵심 수식이 어디서 나왔고 무엇을 뜻하는지 설명한다.\
> 표기: 인라인 수식은 `$...$`, 별도 수식은 `$$...$$` 형태의 표준 Markdown LaTeX로 작성한다.\
> 기준 자료: Nature 본문, Supplementary Information, 공개 Willow 105Q 데이터 분석

## 0. 먼저 알아둘 기호

| 기호 | 뜻 |
|---|---|
| $d$ | Surface-code distance. 오류를 견디는 코드의 크기 |
| $p$ | 물리 연산의 대표 오류율 |
| $p_{\mathrm{thr}}$ | Surface-code threshold 오류율 |
| $p_L(t)$ | $t$ cycle 전체를 실행한 뒤 논리값이 틀릴 확률 |
| $\epsilon_d$ | Distance $d$에서 QEC 한 cycle당 논리 오류율 |
| $\Lambda$ | Distance를 2만큼 키웠을 때 논리 오류가 감소하는 배율 |
| $p_{\mathrm{det}}$ | Stabilizer 비교에서 detector가 반응한 평균 비율 |
| $t_c$ | QEC 한 cycle의 시간 |
| $\Gamma$ | 상태가 시간에 따라 손상되는 유효 속도 |

## 1. Surface code에 필요한 물리 큐비트 수

논문은 일반적인 distance-$d$ surface-code memory의 물리 큐비트 수를 다음과 같이 설명한다.

$$
N_{\mathrm{physical}} = 2d^2-1
$$

Data qubit과 measure qubit의 수를 각각 계산하면 다음과 같다.

$$
\begin{aligned}
N_{\mathrm{data}} &= d^2, \\
N_{\mathrm{measure}} &= d^2-1, \\
N_{\mathrm{physical}}
&= N_{\mathrm{data}}+N_{\mathrm{measure}} \\
&= d^2+(d^2-1) \\
&= 2d^2-1.
\end{aligned}
$$

예를 들어 $d=7$이면:

$$
\begin{aligned}
N_{\mathrm{data}} &= 7^2=49, \\
N_{\mathrm{measure}} &= 7^2-1=48, \\
N_{\mathrm{physical}} &= 49+48=97.
\end{aligned}
$$

Willow $d=7$ 실험은 여기에 경계의 leakage-removal qubit 4개를 추가하여 총 101개를 사용했다.

## 2. Distance에 따른 논리 오류 억제: 논문 식 (1)

논문의 가장 중요한 근사식은 다음과 같다.

$$
\epsilon_d \propto
\left(\frac{p}{p_{\mathrm{thr}}}\right)^{(d+1)/2}
$$

### 2.1 왜 지수가 $(d+1)/2$인가

Distance $d$인 코드는 원칙적으로 다음 개수까지의 오류를 교정할 수 있다.

$$
t_{\mathrm{correct}}
= \left\lfloor\frac{d-1}{2}\right\rfloor
$$

따라서 decoder를 속여 logical error를 만들려면 최소한 다음 정도의 오류가 필요하다. 논문에서 사용한 $d=3,5,7$은 모두 홀수이므로 두 표현이 같다.

$$
t_{\mathrm{fail}}
= t_{\mathrm{correct}}+1
= \left\lceil\frac{d}{2}\right\rceil
= \frac{d+1}{2}
$$

각 물리 오류가 대략 확률 $p$로 발생한다고 생각하면, 필요한 오류가 동시에 모일 확률의 leading term은 다음과 같은 모양이다.

$$
p^{(d+1)/2}
$$

실제로는 오류가 놓일 수 있는 경로의 수, 회로 구조와 decoder 성능도 영향을 준다. 논문 식에서는 이런 복잡한 요소를 threshold $p_{\mathrm{thr}}$와 비례상수 $A$에 포함해 다음 형태로 요약한다.

$$
\epsilon_d \approx
A\left(\frac{p}{p_{\mathrm{thr}}}\right)^{(d+1)/2}
$$

이 식은 정확한 항등식이 아니라 threshold보다 충분히 낮은 영역에서 기대하는 scaling model이다.

### 2.2 Threshold 위아래의 차이

Threshold 아래에서는:

$$
p<p_{\mathrm{thr}}
\quad\Longrightarrow\quad
\frac{p}{p_{\mathrm{thr}}}<1
$$

1보다 작은 수는 지수가 커질수록 더 작아진다. 따라서 $d$가 증가하면 $\epsilon_d$가 감소한다.

Threshold 위에서는:

$$
p>p_{\mathrm{thr}}
\quad\Longrightarrow\quad
\frac{p}{p_{\mathrm{thr}}}>1
$$

이 경우에는 지수가 커질수록 값이 증가해 큰 코드의 이점이 사라진다.

## 3. Suppression factor $\Lambda$의 도출

$\Lambda$는 distance를 $d$에서 $d+2$로 키웠을 때 오류가 감소하는 배율이다.

$$
\Lambda = \frac{\epsilon_d}{\epsilon_{d+2}}
$$

앞의 scaling model을 두 distance에 적용하면:

$$
\begin{aligned}
\epsilon_d
&\approx A\left(\frac{p}{p_{\mathrm{thr}}}\right)^{(d+1)/2}, \\
\epsilon_{d+2}
&\approx A\left(\frac{p}{p_{\mathrm{thr}}}\right)^{(d+3)/2}.
\end{aligned}
$$

두 식을 나누면 $A$가 없어지고 지수 차이는 1이 된다.

$$
\begin{aligned}
\Lambda
&= \frac{\epsilon_d}{\epsilon_{d+2}} \\
&\approx
\left(\frac{p}{p_{\mathrm{thr}}}\right)^{-1} \\
&\approx \frac{p_{\mathrm{thr}}}{p}.
\end{aligned}
$$

따라서 이상적인 단순 모델에서는:

$$
\begin{aligned}
\Lambda>1 &\iff p<p_{\mathrm{thr}}, \\
\Lambda=1 &\iff p=p_{\mathrm{thr}}, \\
\Lambda<1 &\iff p>p_{\mathrm{thr}}.
\end{aligned}
$$

실제 장치에서는 correlation, leakage, finite-size effect와 decoder 때문에 $\Lambda$가 단순히 $p_{\mathrm{thr}}/p$와 정확히 같지는 않다.

## 4. 전체 실패율 $p_L(t)$와 cycle당 오류율 $\epsilon_d$

### 4.1 왜 단순히 $p_L=t\epsilon_d$가 아닌가

한 cycle마다 logical state가 확률 $\epsilon_d$로 뒤집힌다고 가정한다. $t$ cycle 후 최종 논리값이 틀리려면 flip이 홀수 번 일어나야 한다.

- 1번 flip: 실패
- 2번 flip: 원래 값으로 복귀
- 3번 flip: 실패
- 4번 flip: 원래 값으로 복귀

짝수 번 flip 확률을 $P_{\mathrm{even}}$, 홀수 번 flip 확률을 $P_{\mathrm{odd}}$라고 하면:

$$
\begin{aligned}
P_{\mathrm{even}}+P_{\mathrm{odd}} &= 1, \\
P_{\mathrm{even}}-P_{\mathrm{odd}}
&= (1-2\epsilon_d)^t.
\end{aligned}
$$

두 식을 풀면 $P_{\mathrm{odd}}=p_L(t)$이므로:

$$
p_L(t)
= \frac{1-(1-2\epsilon_d)^t}{2}
$$

이 식은 cycle마다 동일한 $\epsilon_d$를 가지며 logical flip이 독립적이라는 유효 모델 아래에서 정확하다.

### 4.2 측정한 $p_L$에서 $\epsilon_d$ 구하기

앞의 식을 $\epsilon_d$에 대해 풀면:

$$
\begin{aligned}
1-2p_L &= (1-2\epsilon_d)^t, \\
(1-2p_L)^{1/t} &= 1-2\epsilon_d, \\
\epsilon_d
&= \frac{1-(1-2p_L)^{1/t}}{2}.
\end{aligned}
$$

논문의 one-point estimate와 본 프로젝트의 짧은 모의실험에서 이 식을 사용한다.

### 4.3 작은 오류율 근사

$\epsilon_d$가 충분히 작으면 이항전개의 첫 항만 남길 수 있다.

$$
(1-2\epsilon_d)^t
\approx 1-2t\epsilon_d
$$

따라서:

$$
\begin{aligned}
p_L(t) &\approx t\epsilon_d, \\
\epsilon_d &\approx \frac{p_L}{t}.
\end{aligned}
$$

다음 항까지 포함하면:

$$
p_L(t)
\approx t\epsilon_d-t(t-1)\epsilon_d^2+\mathcal{O}(\epsilon_d^3)
$$

그러므로 $p_L\approx t\epsilon_d$는 $t\epsilon_d$가 매우 작을 때만 좋은 근사다. 본 프로젝트의 최종 Willow fitting에는 이 단순 근사를 사용하지 않았다.

## 5. 여러 cycle 자료에서 $\epsilon_d$를 fitting하는 방법

정확한 누적 실패식에서 시작한다.

$$
1-2p_L(t)=(1-2\epsilon_d)^t
$$

양변에 자연로그를 취하면:

$$
\ln\!\left[1-2p_L(t)\right]
= t\ln(1-2\epsilon_d)
$$

실제 fitting에서는 cycle 수와 무관한 준비·마지막 측정 효과를 흡수하기 위해 절편 $b$도 허용한다.

$$
\begin{aligned}
y &= b+st, \\
y &= \ln\!\left[1-2p_L(t)\right], \\
s &= \ln(1-2\epsilon_d).
\end{aligned}
$$

직선 기울기 $s$를 얻은 뒤:

$$
\begin{aligned}
e^s &= 1-2\epsilon_d, \\
\epsilon_d &= \frac{1-e^s}{2}.
\end{aligned}
$$

이것이 공개 Willow 데이터에서 각 code 위치와 X/Z basis별 $\epsilon_d$를 구한 방법이다.

### Fitting이 성립하려면

$$
0\le p_L<\frac{1}{2}
$$

이어야 $\ln(1-2p_L)$가 정의된다. 본 프로젝트의 Direct PyMatching 결과 중 late-cycle 측정점 4개는 $p_L\ge 0.5$여서 이 fitting에서만 제외하고 raw CSV에는 보존했다.

## 6. $d=3,5,7$에서 하나의 $\Lambda$를 fitting하는 방법

Distance scaling model을 다음처럼 쓴다.

$$
\epsilon_d
= \frac{A}{\Lambda^{(d+1)/2}}
$$

로그를 취하면:

$$
\ln\epsilon_d
= \ln A-\frac{d+1}{2}\ln\Lambda
$$

다음 좌표를 직선으로 fitting한다.

$$
x=\frac{d+1}{2},
\qquad
y=\ln\epsilon_d
$$

직선 기울기를 $m$이라 하면:

$$
\begin{aligned}
m &= -\ln\Lambda, \\
\Lambda &= e^{-m}.
\end{aligned}
$$

### 구간별 $\Lambda$와 fitted $\Lambda$의 차이

Public Libra 재분석 결과:

$$
\begin{aligned}
\Lambda_{3\rightarrow5}
&= \frac{\epsilon_3}{\epsilon_5}\approx 2.02, \\
\Lambda_{5\rightarrow7}
&= \frac{\epsilon_5}{\epsilon_7}\approx 2.04, \\
\Lambda_{\mathrm{fit}} &\approx 2.029.
\end{aligned}
$$

Direct PyMatching 결과:

$$
\begin{aligned}
\Lambda_{3\rightarrow5} &\approx 1.470, \\
\Lambda_{5\rightarrow7} &\approx 1.583, \\
\Lambda_{\mathrm{fit}} &\approx 1.525.
\end{aligned}
$$

Fitted $\Lambda$는 모든 distance에서 정확히 같은 비율이라는 뜻이 아니라, 조사한 distance 범위의 평균적인 지수 감소 속도다.

논문은 $\epsilon_d$의 불확실성을 포함한 회귀를 사용한다. 본 프로젝트는 distance별 평균 $\epsilon$에 대한 단순 비가중 직선 fitting을 사용하므로 uncertainty 계산은 논문과 완전히 같지 않다.

## 7. Detection probability와 logical error의 power-law fit

논문은 평균 detector 반응 비율 $p_{\mathrm{det}}$를 전체 물리 오류량의 proxy로 사용한다.

$$
p_{\mathrm{det}}
= \frac{\text{반응한 weight-4 detector 비교 수}}
{\text{전체 weight-4 detector 비교 수}}
$$

물리 오류율 $p$가 $p_{\mathrm{det}}$에 대략 비례한다고 가정하면 논문 식 (1)에서:

$$
p\propto p_{\mathrm{det}}
\quad\Longrightarrow\quad
\epsilon_d\propto p_{\mathrm{det}}^{(d+1)/2}
$$

실험에서는 지수를 고정하지 않고 다음 식으로 fitting한다.

$$
\epsilon_d=A_d p_{\mathrm{det}}^k
$$

로그를 취하면:

$$
\ln\epsilon_d
= \ln A_d+k\ln p_{\mathrm{det}}
$$

따라서 log-log graph의 기울기가 $k$다. 이상적인 독립 오류 모델이면:

$$
k_{\mathrm{ideal}}=\frac{d+1}{2}
$$

논문 실험에서는 $k/k_{\mathrm{ideal}}$이 평균 약 0.8이었다. 즉 $p_{\mathrm{det}}$가 감소할 때 logical error가 좋아지기는 했지만, 이상적인 독립 오류 모델이 예측한 속도의 약 80%만 달성했다. 논문은 장치의 correlated error를 주요 원인으로 해석한다.

### 왜 $p_{\mathrm{det}}$가 $1/\Lambda$의 proxy인가

앞서 얻은 관계는:

$$
\frac{1}{\Lambda}\approx\frac{p}{p_{\mathrm{thr}}}
$$

그리고 $p\propto p_{\mathrm{det}}$라고 보면:

$$
\frac{1}{\Lambda}\propto p_{\mathrm{det}}
$$

따라서 $p_{\mathrm{det}}$가 작을수록 $\Lambda$가 커질 것으로 기대한다. 다만 leakage처럼 오류 구조를 바꾸는 현상에서는 같은 $p_{\mathrm{det}}$라도 $\Lambda$가 달라질 수 있다.

## 8. $1/\Lambda$ error budget의 도출

Supplementary Information은 72Q processor의 error budget을 다음처럼 쓴다.

$$
\frac{1}{\Lambda_{3/5}}
= \sum_i w_i p_i
$$

여기서:

- $p_i$: $i$번째 물리 오류 채널의 측정 오류율
- $w_i$: 그 오류율이 $1/\Lambda$를 얼마나 변화시키는지 나타내는 민감도

이 식은 여러 오류가 완전히 독립적으로 더해진다는 근본 법칙이 아니라, 현재 동작점 주변에서 함수를 1차 Taylor 전개한 선형 근사다.

일반적인 1차 전개는:

$$
f(p_1,p_2,\ldots)
\approx f_0+
\sum_i
\frac{\partial f}{\partial p_i}\,\Delta p_i
$$

여기서 $f=1/\Lambda$, 민감도 $w_i\approx\partial(1/\Lambda)/\partial p_i$로 두면 error-budget 식을 얻는다. 이 근사로 각 오류 채널이 전체 suppression 손실에 기여하는 비율을 계산할 수 있다.

## 9. Logical lifetime과 beyond breakeven

논문은 X 또는 Z 한 상태만 유리하게 선택하지 않도록 Bloch sphere의 모든 순수 상태를 평균한 channel fidelity를 사용한다.

개념적인 정의는 다음과 같다.

$$
F[\mathcal{E}]
= \int d\psi\;
F\!\left(
|\psi\rangle\!\langle\psi|,
\mathcal{E}(|\psi\rangle\!\langle\psi|)
\right)
$$

Qubit의 선형성 때문에 실제로는 Bloch sphere의 여섯 축 상태를 사용해 평균할 수 있다.

$$
|{+X}\rangle,\ |{-X}\rangle,\
|{+Y}\rangle,\ |{-Y}\rangle,\
|{+Z}\rangle,\ |{-Z}\rangle
$$

짧은 시간 $\delta t$에서 fidelity 감소를 다음처럼 정의한다.

$$
F(\delta t)
\approx 1-\frac{\Gamma}{2}\delta t
$$

여기서 $\Gamma$는 유효 상태 손상 속도이고 유효 lifetime은:

$$
T_{\mathrm{eff}}=\frac{1}{\Gamma}
$$

### 9.1 물리 큐비트

Amplitude damping 속도를 $\gamma_1$, 전체 dephasing 속도를 $\gamma_2$라고 하면:

$$
\Gamma_{\mathrm{physical}}
= \frac{\gamma_1+2\gamma_2}{3}
$$

보통:

$$
\gamma_1=\frac{1}{T_1},
\qquad
\gamma_2=\frac{1}{T_2}
$$

이 식은 X, Y, Z 방향의 상태가 서로 다른 속도로 손상되는 효과를 균일 평균한 결과다.

### 9.2 Logical qubit

한 cycle의 logical Pauli error 확률을 $p_X,p_Y,p_Z$라고 하면 연속 시간 속도는 대략:

$$
\gamma_X\approx\frac{p_X}{t_c},
\qquad
\gamma_Y\approx\frac{p_Y}{t_c},
\qquad
\gamma_Z\approx\frac{p_Z}{t_c}
$$

Logical channel의 유효 손상 속도는:

$$
\Gamma_{\mathrm{logical}}
= \frac{4}{3}(\gamma_X+\gamma_Y+\gamma_Z)
$$

X/Z memory에서 측정한 평균 $\epsilon_d$와의 관계는:

$$
\begin{aligned}
\epsilon_X &= p_Z+p_Y, \\
\epsilon_Z &= p_X+p_Y, \\
\epsilon_d &= \frac{\epsilon_X+\epsilon_Z}{2}.
\end{aligned}
$$

따라서:

$$
\begin{aligned}
p_X+p_Y+p_Z
&= 2\epsilon_d-p_Y \\
&\le 2\epsilon_d.
\end{aligned}
$$

이를 $\Gamma_{\mathrm{logical}}$ 식에 넣으면:

$$
\Gamma_{\mathrm{logical}}
\le \frac{8\epsilon_d}{3t_c}
$$

### 9.3 Breakeven gain

논문은 가장 좋은 물리 큐비트와 logical qubit의 유효 lifetime 비를 gain $G$로 정의한다.

$$
\begin{aligned}
G
&= \frac{T_{\mathrm{logical}}}{T_{\mathrm{best\ physical}}} \\
&= \frac{1/\Gamma_{\mathrm{logical}}}
{1/\Gamma_{\mathrm{physical}}}.
\end{aligned}
$$

Willow $d=7$ 결과는 다음과 같다.

$$
\begin{aligned}
T_{\mathrm{logical}} &= 291\pm6\ \mu\mathrm{s}, \\
T_{\mathrm{best\ physical}} &= 119\pm13\ \mu\mathrm{s}, \\
G &\approx \frac{291}{119} \\
&\approx 2.45 \\
&\longrightarrow 2.4\pm0.3.
\end{aligned}
$$

$$
G>1
\quad\Longrightarrow\quad
\text{logical memory가 가장 좋은 물리 큐비트보다 오래 유지됨}
$$

이것이 beyond breakeven의 의미다.

## 10. 통계적 불확실성

$N$번 실험에서 logical failure가 이항분포를 따른다고 보면, 측정 실패율 $p_L$의 표준오차는 대략:

$$
\sigma_{p_L}
\approx \sqrt{\frac{p_L(1-p_L)}{N}}
$$

논문은 이 shot noise뿐 아니라 시간에 따른 장치 성능 변화가 만든 추가 잔차도 $\epsilon_d$의 불확실성에 포함한다.

여러 code·basis의 $\epsilon$ 불확실성이 각각 $\sigma_i$일 때 평균 $\epsilon$의 불확실성은:

$$
\sigma_{\mathrm{mean}}
= \frac{\sqrt{\sum_i\sigma_i^2}}{n}
$$

$\Lambda$ fitting에서:

$$
m=-\ln\Lambda,
\qquad
\Lambda=e^{-m}
$$

기울기 불확실성이 $\delta m$이면 작은 불확실성 근사에서:

$$
\delta\Lambda
\approx \Lambda\,\delta m
$$

본 프로젝트의 간이 uncertainty는 회귀에서 얻은 $\epsilon$ 불확실성을 전파했지만, 논문의 device-drift 보정까지 완전히 재현하지는 않았다.

## 11. 우리 프로젝트에서 실제로 사용한 수식

| 수식 | 사용 여부 | 용도 |
|---|---|---|
| $p_L(t)=[1-(1-2\epsilon_d)^t]/2$ | 사용 | 전체 실패율과 cycle당 오류율의 관계 |
| $\epsilon_d=[1-(1-2p_L)^{1/t}]/2$ | 사용 | 모의실험의 cycle당 오류율 계산 |
| $\ln(1-2p_L)=b+st$ | 사용 | 공개 Willow 여러-cycle fitting |
| $\epsilon_d=(1-e^s)/2$ | 사용 | Fitted slope에서 $\epsilon_d$ 계산 |
| $\ln\epsilon_d=\ln A-[(d+1)/2]\ln\Lambda$ | 사용 | $d=3,5,7$에서 fitted $\Lambda$ 계산 |
| $\epsilon_d\propto(p/p_{\mathrm{thr}})^{(d+1)/2}$ | 해석에 사용 | Threshold scaling 설명 |
| $\epsilon_d=A_d p_{\mathrm{det}}^k$ | 논문 리뷰에 사용 | Detection probability sensitivity 설명 |
| $1/\Lambda=\sum_i w_i p_i$ | 논문 리뷰에만 사용 | 논문의 error budget 설명 |
| $\Gamma_{\mathrm{logical}}\le8\epsilon_d/(3t_c)$ | 논문 리뷰에만 사용 | Logical lifetime과 breakeven 설명 |

## 12. 가장 중요한 해석상의 주의점

1. $\epsilon_d$는 물리 큐비트 하나의 오류율이 아니라 decoding 후 logical qubit의 cycle당 오류율이다.
2. $\Lambda$는 조사한 distance 범위의 평균적인 감소 속도이며 모든 구간에서 정확히 같은 값일 필요는 없다.
3. $p_L\approx t\epsilon_d$는 작은 오류율·짧은 시간에서만 쓰는 근사이며 최종 Willow fitting에는 사용하지 않았다.
4. $\epsilon_d\propto(p/p_{\mathrm{thr}})^{(d+1)/2}$는 threshold 아래의 leading-order scaling model이다.
5. 같은 $p_{\mathrm{det}}$라도 correlated error나 leakage가 다르면 logical error와 $\Lambda$가 달라질 수 있다.
6. Public Libra의 2.029는 공개 prediction 재분석 결과이고, Direct PyMatching의 1.525는 우리가 직접 decoding한 결과다.

## 13. 참고 위치

- Nature 본문 식 (1): distance scaling과 $\Lambda\approx p_{\mathrm{thr}}/p$
- Nature Fig. 1c,d: $p_L(t)$, $\epsilon_d$와 $\Lambda$ fitting
- Nature Fig. 2b: detection probability와 logical error power law
- Supplementary Section IV.A.3, Eq. (5): $1/\Lambda$ error budget
- Supplementary Section IV.B, Eqs. (6)–(15): average fidelity, lifetime와 breakeven
- Supplementary Section VI.B: one-point $p_L(t)$–$\epsilon_d$ 변환
- Supplementary Section VI.A: 여러 cycle 및 distance에 대한 $\epsilon_d$와 $\Lambda$ fitting·uncertainty

원문 파일:

- `qec_below_threshold_review/paper/quantum_error_correction_below_surface_code_threshold.pdf`
- `qec_below_threshold_review/paper/supplementary_information.pdf`
