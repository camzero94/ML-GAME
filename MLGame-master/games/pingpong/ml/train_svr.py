import pickle
import numpy as np
from sklearn.svm import SVR




#試取資料
file = open(r"C:\Users\User\Desktop\NCKU\INTRODUCCION-ML\MLGame-master\games\pingpong\log\n1 (1).pickle", "rb")

data = pickle.load(file)
file.close()

game_info = data['ml_1P' ]['scene_info']
game_command = data['ml_1P' ]['command']





for i in range(1, 81):
    path = r"C:\Users\User\Desktop\NCKU\INTRODUCCION-ML\MLGame-master\games\pingpong\log\n1 (" + str(i) + ").pickle"
    file = open(path, "rb")
    data = pickle.load(file)
    game_info = game_info + data['ml_1P']['scene_info']
    game_command = game_command + data['ml_1P']['command']
    file.close()
    
print(len(game_info))
print(len(game_command))

X = [None] * len(game_info)
y = [100] * len(game_info)

save_index = 0

for current_frame_index, frame_info in enumerate(game_info):
    # If just hit the ball set labels and set new index

    if frame_info["ball"][1] == 415:
        for j in range(save_index, current_frame_index  + 1):
            y[j] = frame_info["ball"][0] if frame_info["ball"][0] != None else 100
        save_index = current_frame_index
    
    X[current_frame_index] = [
        frame_info["ball"][0], 
        frame_info["ball"][1],
        frame_info["ball_speed"][0],
        frame_info["ball_speed"][1],
        frame_info["blocker"][0]]
        
X_np = np.array(X, dtype=np.float32)
y_np = np.array(y,dtype=np.float32)

model = SVR( C = 1.0, epsilon = 0.2).fit(X_np, y_np)



