import ALL_Files
import motor_steering
def Penguin(direction:str):
	if direction== 'left':
		motor_steering.set_motor_speeds(-10)
	elif direction=='right':
		motor_steering.set_motor_speeds(10)
def motor_turn(turn_angle:int):
	#Using dot turn
	pass
        #Create function with testing.
def main(angle:degrees):
	if (angle>45) or (angle<-45):
		motor_turn(angle/2)
	elif angle>0:
		Penguin(left)
	elif angle<0:
		Penguin(right)
	else:
		pass
