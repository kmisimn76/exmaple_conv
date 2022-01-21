# 제목 없음 

# Convolution kernels 

## conv_k1s1_* 

- Weight(kernel) size가 1x1 인 convolution 커널 변형 
- 코드 기본형은 아래와 같음 

```python 
for M: 
for W: 
for H: 
for C: 
output[M][H][H] += weight[M][C] * input[C][H][W] 
``` 

- OpenCL GPU 특성에 따라 기본적으로 입력은 C, 출력은 M, weight는 M 방향으로 4개씩 벡터화되어 아래와 같이 연산 (벡터는 볼드처리) 

```python 
for M/4: 
for W: 
for H: 
for C/4: 
**output[M][H][H]** += **weight[M][C*4+0]** * (vec4)input[C][H][W][0] 
**output[M][H][H]** += **weight[M][C*4+1]** * (vec4)input[C][H][W][1] 
**output[M][H][H]** += **weight[M][C*4+2]** * (vec4)input[C][H][W][2] 
**output[M][H][H]** += **weight[M][C*4+3]** * (vec4)input[C][H][W][3] 
``` 

- 파일명 중 wchw가 있는 것은, C/M이 아니라 W방향으로 벡터화한 것 
- 파일명 중 숫자(211 등)는 thread 1개가 처리하는 M,H,W의 개수. 211은 thread 1개가 2개의 M 출력을 처리함. 

```python 
for M**/2**: 
for W: 
for H: 
for C: 
output[M*2+0][H][H] += weight[M*2+0][C] * input[C][H][W] 
output[M*2+1][H][H] += weight[M*2+1][C] * input[C][H][W] 
``` 

- 명확히 C,H,W에 대해 숫자가 쓰여 있는 부분도 위와 같음 
- 파일명에 is가 있는 것은 sparsity 관련한 것으로 무시해도 됨
