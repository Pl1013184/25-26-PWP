import ALL_Files

def motor_turn():
	pass
        #Create function with testing.
def main(angle:degrees):
	if (angle>45) or (angle<-45):
		motor_turn(angle/2)
