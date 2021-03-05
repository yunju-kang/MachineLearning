from sklearn import svm, metrics
import random, re


''''''''''''''''''''''''''''''''''''''''''
# csv 데이터 저장
''''''''''''''''''''''''''''''''''''''''''


## csv 변수
csv = []

## csv 파일 불러오기
with open('iris.csv', 'r', encoding = 'utf-8') as fp:



    ### 데이터 정제
    for line in fp:


        #### 공백 줄 제거
        line = line.strip()

        #### 쉼표 기준으로 자르기
        cols = line.split(',')

        #### 문자열 형식의 숫자 데이터를 실수로 저장
        fn = lambda n : float(n) if re.match(r'^[0-9/.]+$', n) else n
        cols = list(map(fn, cols))


        ### 정제한 데이터 저장
        csv.append(cols)



''''''''''''''''''''''''''''''''''''''''''
# csv 데이터 정제
''''''''''''''''''''''''''''''''''''''''''



## 헤더 삭제
del csv[0]



## 데이터 섞기
random.shuffle(csv)



## 학습, 시험 전용 데이터 설정
total_len = len(csv)


### 변수 설정
train_data = []
train_label = []
test_data = []
test_label = []

train_len = total_len * 2/3



for i in range(total_len):


    #### data 설정
    data = csv[i][0:3]
    #### label 설정
    label = csv[i][4]


    if i <= train_len:
        #### 학습 전용 데이터
        train_data.append(data)
        train_label.append(label)


    else:
        #### 시험 전용 데이터
        test_data.append(data)
        test_label.append(label)