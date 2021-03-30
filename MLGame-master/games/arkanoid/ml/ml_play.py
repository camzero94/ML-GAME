
"""
The template of the main script of the machine learning process
"""
import random
import os.path
import pickle

class MLPlay:
    def __init__(self):
        """
        Constructor
        """
        filename = 'model.pickle'
        filepath = os.path.join(os.path.dirname(__file__), filename)
        self.model = pickle.load(open(filepath, 'rb'))
        self.ball_served = False

    def update(self, scene_info):
      #  print (len(scene_info["bricks"]))
        """
        Generate the command according to the received `scene_info`.
        """
        # Make the caller to invoke `reset()` for the next round.
        if (scene_info["status"] == "GAME_OVER" or
            scene_info["status"] == "GAME_PASS"):
            return "RESET"

        if not self.ball_served and len(scene_info["bricks"]) >93 :
            
            if scene_info["platform"][0] > 90:
                self.ball_served = True
                command = "SERVE_TO_LEFT"
            else:
                command = "MOVE_RIGHT"
                
        else:
            nx = scene_info["ball"][0]
            ny = scene_info["ball"][1]
            px = scene_info["platform"][0]
            command = self.model.predict([[nx, ny, px]])

            if command == 0: return "NONE"
            elif command == 1: return "MOVE_LEFT"
            else: command = "MOVE_RIGHT"
        
        return command

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False

##"""
##The template of the main script of the machine learning process
##"""
##import os
##import pickle
##
##import numpy as np
##
### python MLGame.py -i model_DJ.py arkanoid NORMAL 2
##class MLPlay:
##    def __init__(self):
##        """
##        Constructor
##        """
##        self.ball_served = False
##        self.previous_ball = (0, 0)
##        # Need scikit-learn==0.22.2 
##        # with open(os.path.join(os.path.dirname(__file__), 'save', 'model.pickle'), 'rb') as f:
##        #     self.model = pickle.load(f)
##        with open(os.path.join(os.path.dirname(__file__), 'my_model.pickle(3)'), 'rb') as f:
##            self.model = pickle.load(f)
##
##    def update(self, scene_info):
##        """
##        Generate the command according to the received `scene_info`.
##        """
##        # Make the caller to invoke `reset()` for the next round.
##        if (scene_info["status"] == "GAME_OVER" or
##                scene_info["status"] == "GAME_PASS"):
##            return "RESET"
##
##        if not self.ball_served:
##            self.ball_served = True
##            command = "SERVE_TO_LEFT"
##        else:
##            Ball_x = scene_info["ball"][0]
##            Ball_y = scene_info["ball"][1]
##            Ball_speed_x = scene_info["ball"][0] - self.previous_ball[0]
##            Ball_speed_y = scene_info["ball"][1] - self.previous_ball[1]
##            Platform = scene_info["platform"][0]
##            if Ball_speed_x > 0:
##                if Ball_speed_y > 0:
##                    Direction = 0
##                else:
##                    Direction = 1
##            else:
##                if Ball_speed_y > 0:
##                    Direction = 2
##                else:
##                    Direction = 3
##            x = np.array([Ball_x, Ball_y, Direction, Ball_speed_x, Ball_speed_y, Platform]).reshape((1, -1))
##            y = self.model.predict(x)
##
##            if y == 0:
##                command = "NONE"
##            elif y == -1:
##                command = "MOVE_LEFT"
##            elif y == 1:
##                command = "MOVE_RIGHT"
##
##        self.previous_ball = scene_info["ball"]
##        return command
##
##    def reset(self):
##        """
##        Reset the status
##        """
##        self.ball_served = False
##



#import math
#"""
#The template of the main script of the machine learning process
#"""
#
#class MLPlay:
#    def __init__(self):
#        """
#        Constructor
#        """
#        self.ball_pos = [93,93]
#        self.ball_served = False
#
#    def update(self, scene_info): 
#    
#        ballPos = self.ball_pos
#        
#        """
#        Generate the command according to the received `scene_info`.
#        """
#        command = "NONE"
#        # Make the caller to invoke `reset()` for the next round.
#        if (scene_info["status"] == "GAME_OVER" or
#            scene_info["status"] == "GAME_PASS"):
#            return "RESET"
#
#        if not self.ball_served:
#            self.ball_served = True
#            command = "SERVE_TO_RIGHT"    
#
#        if (scene_info["ball"][1] - ballPos[1] != 0 and ballPos[1] > 150):
#            xVal = ballPos[0] + ((400 - ballPos[1]) * ((scene_info["ball"][0] - ballPos[0]) / (scene_info["ball"][1] - ballPos[1])))
#            xVal = math.floor(xVal)
#            if(xVal < 0):
#                xVal = 0 - xVal
#            if(xVal // 200) % 2 == 0:
#                xVal = xVal % 200
#            else:
#                xVal = 200 - xVal % 200
#        else:
#            xVal = 100
#        
#        self.ball_pos = [scene_info["ball"][0], scene_info["ball"][1] ]
#
#        if scene_info["platform"][0] == xVal:
#            command = "NONE"
#        if scene_info["platform"][0] > xVal - 20:
#            command = "MOVE_LEFT"
#        if scene_info["platform"][0] < xVal - 20:
#            command = "MOVE_RIGHT"
#        
#
#        return command 
#        
#    def reset(self):
#        """
#        Reset the status
#        """
#        self.ball_served = False