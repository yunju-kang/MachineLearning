from sklearn import svm, metrics
import random, re
import pandas as pd
from sklearn.model_selection import train_test_split


# ''''''''''''''''''''''''''''''''''''''''''
# # csv 데이터 저장 - version 1
# ''''''''''''''''''''''''''''''''''''''''''
#
#
# ## csv 변수
# csv = []
#
# ## csv 파일 불러오기
# with open('iris.csv', 'r', encoding = 'utf-8') as fp:
#
#
#
#     ### 데이터 정제
#     for line in fp:
#
#
#         #### 공백 줄 제거
#         line = line.strip()
#
#         #### 쉼표 기준으로 자르기
#         cols = line.split(',')
#
#         #### 문자열 형식의 숫자 데이터를 실수로 저장
#         fn = lambda n : float(n) if re.match(r'^[0-9/.]+$', n) else n
#         cols = list(map(fn, cols))
#
#
#         ### 정제한 데이터 저장
#         csv.append(cols)
#
#
# ''''''''''''''''''''''''''''''''''''''''''
# # csv 데이터 정제 - version 1
# ''''''''''''''''''''''''''''''''''''''''''
#
#
#
# ## 헤더 삭제
# del csv[0]
#
#
#
# ## 데이터 섞기
# random.shuffle(csv)
#
#
#
# ## 학습, 시험 전용 데이터 설정
# total_len = len(csv)
#
#
# ### 변수 설정
# train_data = []
# train_label = []
# test_data = []
# test_label = []
#
# train_len = total_len * 2/3
#
#
#
# for i in range(total_len):
#
#
#     #### data 설정
#     data = csv[i][0:3]
#     #### label 설정
#     label = csv[i][4]
#
#
#     if i <= train_len:
#         #### 학습 전용 데이터
#         train_data.append(data)
#         train_label.append(label)
#
#
#     else:
#         #### 시험 전용 데이터
#         test_data.append(data)
#         test_label.append(label)
#
#

''''''''''''''''''''''''''''''''''''''''''
# csv 데이터 저장 - version 2
''''''''''''''''''''''''''''''''''''''''''

csv = pd.read_csv('iris.csv')


''''''''''''''''''''''''''''''''''''''''''
# csv 데이터 정제 - version 2
''''''''''''''''''''''''''''''''''''''''''

## 필요한 열 추출
csv_data = csv[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
csv_label = csv["variety"]


## train,test 전용 데이터 구분
train_data, test_data, train_label, test_label = \
train_test_split(csv_data, csv_label)


''''''''''''''''''''''''''''''''''''''''''
# 학습
''''''''''''''''''''''''''''''''''''''''''

clf = svm.SVC()
clf.fit(train_data, train_label)


''''''''''''''''''''''''''''''''''''''''''
# 예측
''''''''''''''''''''''''''''''''''''''''''

pre = clf.predict(test_data)
print(test_label)


''''''''''''''''''''''''''''''''''''''''''
# 정답률 구하기
''''''''''''''''''''''''''''''''''''''''''

## 정답률
accuracy = metrics.accuracy_score(test_label, pre)

print("정답률 = ", accuracy)


''' shuffle 때문에 코드 실행마다 예측값 및 정답률이 변화함 '''