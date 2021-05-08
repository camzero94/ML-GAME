import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import  classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit




#試取資料
file = open(r"C:\Users\User\Desktop\NCKU\INTRODUCCION-ML\MLGame-master\games\pingpong\log\n2 (1).pickle", "rb")
data = pickle.load(file)
file.close()



#print(data)

game_info = data['ml_2P' ]['scene_info']
game_command = data['ml_2P' ]['command']
#print(game_info)
#print(game_command)

for i in range(1, 51):
    path = r"C:\Users\User\Desktop\NCKU\INTRODUCCION-ML\MLGame-master\games\pingpong\log\n2 (" + str(i) + ").pickle"
    file = open(path, "rb")
    data = pickle.load(file)
    game_info = game_info + data['ml_2P']['scene_info']
    game_command = game_command + data['ml_2P']['command']
    file.close()
    
print(len(game_info))
print(len(game_command))
g = game_info[1]
#print (game_info)

feature = np.array([g['ball'][0], g['ball'][1],g['ball_speed'][0],g['ball_speed'][1], g['platform_2P'][0]])
#print(feature)

#print(game_command[1])
game_command[1] = 0
#
for i in range(2, len(game_info) - 1):
    g = game_info[i]
    feature = np.vstack((feature, [g['ball'][0], g['ball'][1],g['ball_speed'][0],g['ball_speed'][1], g['platform_2P'][0]]))
    if game_command[i] == "NONE": game_command[i] = 0
    elif game_command[i] == "MOVE_LEFT": game_command[i] = 1
    else: game_command[i] = 2
    
answer = np.array(game_command[1:-1])

#print(feature)
#print(feature.shape)
#print(answer)
#print(answer.shape)

#資料劃分
x_train, x_test, y_train, y_test = train_test_split(feature, answer, test_size=0.3, random_state=9)
#參數區間
param_grid = {'n_neighbors':[1, 2, 3,4,5]}
#交叉驗證 
cv = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=12)
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, verbose=10, n_jobs=-1) #n_jobs為平行運算的數量
grid.fit(x_train, y_train)
grid_predictions = grid.predict(x_test)

#儲存
file = open('model_2P.pickle', 'wb')
pickle.dump(grid, file)
file.close()

#最佳參數
print(grid.best_params_)
#預測結果
#print(grid_predictions)
#混淆矩陣
print(confusion_matrix(y_test, grid_predictions))
#分類結果
print(classification_report(y_test, grid_predictions))

