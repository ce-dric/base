Supervised Learning 은 입력 데이터(input)와 정답 데이터(label) 간 관계를 찾는 방법이라면, <br>
Un-Supervised Learning은 label이 없는 채로 input을 예측하는 것이다. <br>
<br>
이상 탐지(Anomaly Detection) 분야에서 오토 인코더(Auto Encoder)는 입력도 input, 정답도 label이다. <br>
단, 근사치로 구하기에 똑같은 출력을 내기에는 힘들다. <br>
따라서, 얼마나 복원했는지에 대한 오차(복원오차) 혹은 정보 손실값(reconstuction loss)이라는 용어를 사용한다. <br>