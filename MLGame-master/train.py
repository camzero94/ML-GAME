
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import  classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

#試取資料
file = open(r"C:\Users\User\Desktop\NCKU\INTRODUCCION-ML\MLGame-master\games\arkanoid\log\n1 (1).pickle", "rb")
data = pickle.load(file)
file.close()

#print(data)

game_info = data['ml']['scene_info']
game_command = data['ml']['command']
#print(game_info)
#print(game_command)

for i in range(1, 36):
    path = r"C:\Users\User\Desktop\NCKU\INTRODUCCION-ML\MLGame-master\games\arkanoid\log\n1 (" + str(i) + ").pickle"
    file = open(path, "rb")
    data = pickle.load(file)
    game_info = game_info + data['ml']['scene_info']
    game_command = game_command + data['ml']['command']
    file.close()
    
print(len(game_info))
print(len(game_command))

g = game_info[1]

feature = np.array([g['ball'][0], g['ball'][1], g['platform'][0]])
print(feature)

print(game_command[1])
game_command[1] = 0

for i in range(2, len(game_info) - 1):
    g = game_info[i]
    feature = np.vstack((feature, [g['ball'][0], g['ball'][1], g['platform'][0]]))
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
param_grid = {'n_neighbors':[1, 2, 3]}
#交叉驗證 
cv = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=12)
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, verbose=10, n_jobs=-1) #n_jobs為平行運算的數量
grid.fit(x_train, y_train)
grid_predictions = grid.predict(x_test)

#儲存
file = open('arkanoid_n3_20210309_knn_model.pickle', 'wb')
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

#
#"""
#Extraction data form pickle files
#"""
#
#
#file_path = r'C:\Users\User\Desktop\NCKU\INTRODUCCION-ML\MLGame-master\games\arkanoid\log\n1 (1).pickle'
#with open(file_path, 'rb') as f:
#    data = pickle.load(f)
#
#scene_info = data['ml']['scene_info']
#command = data['ml']['command']
#
#Ball_x = []
#Ball_y = []
#Ball_speed_x = []
#Ball_speed_y = []
#Direction = []
#Platform = []
#Command = []
#
#for i, s in enumerate(scene_info[1:-2]):
#    Ball_x.append(s['ball'][0])
#    Ball_y.append(s['ball'][1])
#    Platform.append(s['platform'][0])
#    Ball_speed_x.append(scene_info[i+2]["ball"][0] - scene_info[i+1]["ball"][0])
#    Ball_speed_y.append(scene_info[i+2]["ball"][1] - scene_info[i+1]["ball"][1])
#    if Ball_speed_x[-1] > 0:
#        if Ball_speed_y[-1] > 0:
#            # 右下
#            Direction.append(0)
#        else:
#            # 右上
#            Direction.append(1)
#    else:
#        if Ball_speed_y[-1] > 0:
#            # 左下
#            Direction.append(2)
#        else:
#            # 左上
#            Direction.append(3)
#            
#for c in command[1:-2]:
#    if c == "NONE":
#        Command.append(0)
#    elif c == "MOVE_LEFT":
#        Command.append(-1)
#    elif c == "MOVE_RIGHT":
#        Command.append(1)
#
#numpy_data = np.array([Ball_x, Ball_y, Ball_speed_x, Ball_speed_y, Direction, Platform])
#x = np.transpose(numpy_data) 
#y = command
#
##print (x,y)
#
#scene_info = data['ml']['scene_info']
#command = data['ml']['command']
#
#k = range(1, len(scene_info)-1)
#
#ball_x = np.array([scene_info[i]['ball'][0] for i in k])
#ball_y = np.array([scene_info[i]['ball'][1] for i in k])
#ball_speed_x = np.array([scene_info[i+1]['ball'][0] - scene_info[i]['ball'][0] for i in k])
#ball_speed_y = np.array([scene_info[i+1]['ball'][1] - scene_info[i]['ball'][1] for i in k])
#direction = np.where(np.vstack((ball_speed_x, ball_speed_y)) > 0, [[1],[0]], [[2],[3]]).sum(axis=0)  # x y: ++1, +-4, -+2, --3
#platform = np.array([scene_info[i]['platform'][0] for i in k])
#target = np.where(np.array(command) == 'NONE', 0,
#                  np.where(np.array(command) == 'MOVE_LEFT', -1, 1))[1:-1]  # [0] SERVE_TO_RIGHT, [1897] None
#
#
#x = np.hstack((ball_x.reshape(-1, 1),
#               ball_y.reshape(-1, 1),
#               ball_speed_x.reshape(-1, 1),
#               ball_speed_y.reshape(-1, 1),
#               direction.reshape(-1, 1),
#               platform.reshape(-1, 1)))
#y = target
#w = '\n'
#print (x,3*w,y)
#
## train data
#from sklearn.neighbors import KNeighborsClassifier
#
#model = KNeighborsClassifier(n_neighbors=3)
#print(model.fit(x, y))
#print(model.score(x, y))
#
#with open('extract.pickle', 'wb') as f:
#    pickle.dump(model, f)