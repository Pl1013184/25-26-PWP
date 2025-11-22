import cv2 as cv
import numpy as np
class cvcol():
	def __init__(guy,path,gray=False):
		if gray:
			guy.img=cv.imread(path, 0)
		else:
			guy.img=cv.imread(path)
	def dialate(guy):
		guy.img=cv.dilate(guy.img,(5,5),iterations=1)
	def erode(guy):
		guy.img=cv.erode(guy.img,(5,5),iterations=1)
	def bigger(guy,factor):
		guy.img=cv.resize(guy.img,(guy.img.shape[1]*factor,guy.img.shape[0]*factor))
	def shower(guy,name="NO NAME? WHAT A SHAME",time=30):
		cv.imshow(name,guy.img)
		cv.waitKey(time*1000)
		cv.destroyAllwindows
	def bluredlines(guy):
		guy.img=cv.GaussianBlur(guy.img,(5,5),0)
	def ret(guy):
		return guy.img