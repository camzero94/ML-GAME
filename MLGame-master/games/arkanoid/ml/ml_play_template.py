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
        ballPos = self.ball_pos
        print (scene_info["ball"])
        """
        Generate the command according to the received `scene_info`.
        """
        command = "NONE"
        # Make the caller to invoke `reset()` for the next round.
        if (scene_info["status"] == "GAME_OVER" or
            scene_info["status"] == "GAME_PASS"):
            return "RESET"

        if not self.ball_served:
            self.ball_served = True
            command = "SERVE_TO_LEFT"    

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
        
        print(command)
        return command 
        # if scene_info["frame"]==0:
        #     current_ball_x = scene_info["ball"][0]
        #     current_ball_y = scene_info["ball"][1]
        #     des_x = 100
        
        # else:
        #     last_ball_x = current_ball_x
        #     last_ball_y = current_ball_y
        #     current_ball_x=scene_info["ball"][0]
        #     current_ball_y=scene_info["ball"][1]

        #     if current_ball_y>last_ball_y:
        #         if current_ball_x>last_ball_x:
        #            des_x=(400-current_ball_y)+current_ball_x	
        #         else:
        #            des_x=current_ball_x-(400-current_ball_y)
        #     if current_ball_y<last_ball_y:
        #         des_x=100

        # while des_x>200 or des_x<0:
            
        #     if des_x>200:
        #         des_x=(200-(des_x-200))
        #     else:
        #         des_x=-des_x
					
		
		
		# ## 3.4. Send the instruction for this frame to the game process
        # if des_x<scene_info["platform"][0]+25:
        #     command = "MOVE_LEFT"
        # elif des_x>scene_info["platform"][0]+25:
        #     command = "MOVE_RIGHT"
        # else:
        #     command = "NONE"	            

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False