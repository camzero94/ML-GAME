import math

"""
The template of the script for the machine learning process in game pingpong
"""

class MLPlay:

    def __init__(self, side):
        """
        Constructor

        @param side A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side.
        """
        self.ball_served = False
        self.side = side
        self.ball_pos = [100,100]
        self.xVal = 0


    def update(self, scene_info):
        """
        Generate the command according to the received scene information
        """
        xVal = self.xVal
        s = self.ball_pos
        side = self.side
        #print ("1p", scene_info["platform_1P"][0])
        #print ("2p", scene_info["platform_2P"][0])
        # s[1] = 93 if side == "1P" else 400
        
        y = scene_info["platform_" + side] [1] + (0 if side == "1P" else 30)
        yEnemy = scene_info["platform_" + ("1P" if side == "2P" else "2P")][1] + (0 if side == "2P" else 30)

        # If ball is coming the previous distance is larger than the new one
        ball_coming = abs(scene_info["ball"][1] - y) <= abs(s[1] - y)
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        
        if not self.ball_served:
            self.ball_served = True
            return "SERVE_TO_RIGHT"
        
        if ((scene_info["ball"][1] - s[1]) != 0 and  (scene_info["ball"][0] - s[0]) != 0):
            m = ((scene_info["ball"][1] - s[1]) / (scene_info["ball"][0] - s[0]))
            if ball_coming:
            #Linear ecuation (Two points to predict the position in x)
                xVal = s[0] + ((y-s[1]) / m)
                xVal = math.floor(xVal)
             #   print("Frame:",scene_info["frame"],"Speed: " ,scene_info["ball_speed"][1])
              #  print (ball_coming)
                #wall Hitting
                if xVal < 0:
                    xVal = 0 - xVal
                if (xVal//200) % 2 == 0:
                    xVal = xVal % 200
                else: 
                    xVal = 200 - xVal % 200
            else:
                xVal = 100
               
        else:
            xVal = 100
    
    #    if side == "1P":
    #        scene_info['xValue_1P']= xVal
    #    else: 
    #        
    #        scene_info['xValue_2P']= xVal
    #        
    #    print (side,  scene_info["xValue_1P"])
    #    print (side,  scene_info["xValue_2P"])
    
       # print("Frame:",scene_info["frame"],"Speed: " ,scene_info["ball_speed"][1])
       # print (ball_coming)
       # #wall Hitting
       # if xVal < 0:
       #     xVal = 0 - xVal
       # if (xVal//200) % 2 == 0:
       #     xVal = xVal % 200
       # else: 
       #     xVal = 200 - xVal % 200

        
       # print(scene_info["ball"][0],xVal)
        self.ball_pos = scene_info["ball"]
        if scene_info["platform_" + side][0] == xVal - 20:
            return "NONE"
        if scene_info["platform_" + side][0] > xVal - 20:
            return "MOVE_LEFT"
        if scene_info["platform_" + side][0] < xVal - 20:
            return "MOVE_RIGHT"
  


    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False
