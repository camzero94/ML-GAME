import pickle
import numpy as np

file_path = r'C:\Users\User\Desktop\NCKU\INTRODUCCION-ML\MLGame-master\games\arkanoid\log\n1 (1).pickle'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

scene_info = data['ml']['scene_info']
command = data['ml']['command']

Ball_x = []
Ball_y = []
Ball_speed_x = []
Ball_speed_y = []
Direction = []
Platform = []
Command = []

for i, s in enumerate(scene_info[1:-2]):
    Ball_x.append(s['ball'][0])
    Ball_y.append(s['ball'][1])
    Platform.append(s['platform'][0])
    Ball_speed_x.append(scene_info[i+2]["ball"][0] - scene_info[i+1]["ball"][0])
    Ball_speed_y.append(scene_info[i+2]["ball"][1] - scene_info[i+1]["ball"][1])
    if Ball_speed_x[-1] > 0:
        if Ball_speed_y[-1] > 0:
            # 右下
            Direction.append(0)
        else:
            # 右上
            Direction.append(1)
    else:
        if Ball_speed_y[-1] > 0:
            # 左下
            Direction.append(2)
        else:
            # 左上
            Direction.append(3)
            
for c in command[1:-2]:
    if c == "NONE":
        Command.append(0)
    elif c == "MOVE_LEFT":
        Command.append(-1)
    elif c == "MOVE_RIGHT":
        Command.append(1)

numpy_data = np.array([Ball_x, Ball_y, Ball_speed_x, Ball_speed_y, Direction, Platform])
x = np.transpose(numpy_data) 
y = command

#print (x,y)

scene_info = data['ml']['scene_info']
command = data['ml']['command']

k = range(1, len(scene_info)-1)

ball_x = np.array([scene_info[i]['ball'][0] for i in k])
ball_y = np.array([scene_info[i]['ball'][1] for i in k])
ball_speed_x = np.array([scene_info[i+1]['ball'][0] - scene_info[i]['ball'][0] for i in k])
ball_speed_y = np.array([scene_info[i+1]['ball'][1] - scene_info[i]['ball'][1] for i in k])
direction = np.where(np.vstack((ball_speed_x, ball_speed_y)) > 0, [[1],[0]], [[2],[3]]).sum(axis=0)  # x y: ++1, +-4, -+2, --3
platform = np.array([scene_info[i]['platform'][0] for i in k])
target = np.where(np.array(command) == 'NONE', 0,
                  np.where(np.array(command) == 'MOVE_LEFT', -1, 1))[1:-1]  # [0] SERVE_TO_RIGHT, [1897] None


x = np.hstack((ball_x.reshape(-1, 1),
               ball_y.reshape(-1, 1),
               ball_speed_x.reshape(-1, 1),
               ball_speed_y.reshape(-1, 1),
               direction.reshape(-1, 1),
               platform.reshape(-1, 1)))
y = target
w = '\n'
print (x,3*w,y)

# train data
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
print(model.fit(x, y))
print(model.score(x, y))

with open('extract.pickle', 'wb') as f:
    pickle.dump(model, f)