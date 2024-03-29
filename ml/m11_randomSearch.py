import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# GridSearchCV = RandomizedSearchCV

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding="utf-8")

# 붗꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:,["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

# 그리드 서치에서 사용할 매개 변수 === 1
parameters = {
    "C": [1, 10, 100, 1000], "kernel":["linear"],
    "C": [1, 10, 100, 1000], "kernel":["rbf"], "gamma":[0.001, 0.0001],
    "C": [1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]
}
'''
최적의 매개 변수 =  SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,     
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='sigmoid',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
최종 정답률 =  0.9666666666666667
최종 정답률 =  0.9666666666666667
'''
# 그리드 서치 === 2
kfold_cv = KFold(n_splits=5, shuffle=True)
clf = c
clf.fit(x_train, y_train)
print("최적의 매개 변수 = ", clf.best_estimator_)

# 최적의 매개 변수로 평가하기 === 3
y_pred = clf.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred))
last_score = clf.score(x_test, y_test)
print("최종 정답률 = ", last_score)