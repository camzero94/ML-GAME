import math
"""
The template of the main script of the machine learning process
"""

class MLPlay:
    def __init__(self):
        """
        Constructor
        """
        self.ball_pos = [93,93]
        self.ball_served = False
       

    def update(self, scene_info): 
      
     #  print (scene_info["bricks"][7][1])
        #print(len(scene_info["bricks"]))
        ballPos = self.ball_pos
    
        """
        Generate the command according to the received `scene_info`.
        """
        command = "NONE"
        # Make the caller to invoke `reset()` for the next round.
        if (scene_info["status"] == "GAME_OVER" or
            scene_info["status"] == "GAME_PASS"):
            return "RESET"

        # if not self.ball_served:
        #     self.ball_served = True
        #     command = "SERVE_TO_RIGHT" 
           
        if not self.ball_served :
            if len(scene_info["bricks"]) >93 :
                if scene_info["platform"][0] > 90:
                    self.ball_served = True
                    command = "SERVE_TO_LEFT"
                else:
                    command = "MOVE_RIGHT"
                    return command
    


        if (scene_info["ball"][1] - ballPos[1] != 0 and ballPos[1] > 150):
            xVal = ballPos[0] + ((400 - ballPos[1]) * ((scene_info["ball"][0] - ballPos[0]) / (scene_info["ball"][1] - ballPos[1])))
            xVal = math.floor(xVal)
            if(xVal < 0):
                xVal = 0 - xVal
            if(xVal // 200) % 2 == 0:
                xVal = xVal % 200
            else:
                xVal = 200 - xVal % 200
        else:
            xVal = 100
        
        self.ball_pos = [scene_info["ball"][0], scene_info["ball"][1] ]

        if scene_info["platform"][0] == xVal:
            command = "NONE"
        if scene_info["platform"][0] > xVal - 20:
            command = "MOVE_LEFT"
        if scene_info["platform"][0] < xVal - 20:
            command = "MOVE_RIGHT"
        

        return command 
        
    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False