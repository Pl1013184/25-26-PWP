from flask import Flask, Response,request,render_template
import cv2
import cv2 as cv
import numpy as np
import threading
import time
from threading import Thread
import pi_client
app = Flask(__name__)
global videio
global bholdr
bholdr=[None]*5
print(bholdr)
lock=threading.Lock()
videio = cv2.VideoCapture(0)
print("Width:", videio.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", videio.get(cv2.CAP_PROP_FRAME_HEIGHT))
videio.set(cv.CAP_PROP_FRAME_WIDTH,640)
videio.set(cv.CAP_PROP_FRAME_HEIGHT,480)
#videio.set(cv.CAP_PROP_FPS,30)
#while videio.read()[0]==False:
#   pass
print("Frames Per Second",videio.get(cv.CAP_PROP_FPS))
def gen_frm():
    global bholdr
    i=0
    while True:
        x,raw=videio.read()
        lock.acquire()
        #1
        if not x:
            lock.release()
            continue
        if raw is None:
            lock.release()
            continue
        bholdr[i]=raw
        lock.release()
        i+=1
        if i==4:
            i=0
t=Thread(target=gen_frm,daemon=True)
raw=None
def sumofc(list):
    len_=[]
    ret=list[0]
    for i in list:
        len_.append(len(i))
    for i in list[1:-1]:
        ret+=i[0:min(len_),:,:]
    return ret
t.start()
#while raw is None:
#  pass

def gen_vid():
  global bholdr
 # print(type(bholdr[1]))
  while True:
       lock.acquire()
       if bholdr[3] is None:
           lock.release()
           continue
       raw1=bholdr[3].copy()
       lock.release()
       raw1=cv.resize(raw1,(640,480))
      # print('thought')
       x, img_data = cv2.imencode('.jpg', raw1)
       raw_bytes = img_data.tobytes()
       time.sleep(0.01)
       yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + raw_bytes + b'\r\n')


def gen_proc_vid():
  while True:
      lock.acquire()
      if bholdr[3] is None:
          lock.release()
          continue
      og=bholdr[3].copy()        
      lock.release()
     # raw1=cv.adaptiveThreshold(raw1,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
     # lines = cv.HoughLinesP(raw1,rho=1,theta=np.pi/180,threshold=100 ,minLineLength=100,maxLineGap=12)
     # imag = raw1
      frame = og.copy()
      #rows,cols,ch = og.shape
     # pts1 = np.float32([[50,50],[200,50],[50,200]])
     # pts2 = np.float32([[10,100],[200,50],[100,250]])
     # M = cv.getAffineTransform(pts1,pts2)
     # og = cv.warpAffine(og,M,(cols,rows))
      dil=cv.getStructuringElement(cv.MORPH_RECT, (5,5))
      og=cv.cvtColor(og,cv.COLOR_BGR2HSV)
      h,s,v=cv.split(og)
      v[:,:]=255
      og=cv.merge((h,s,v))
      og=cv.cvtColor(og,cv.COLOR_HSV2BGR)
      og=cv.cvtColor(og,cv.COLOR_BGR2GRAY)
      og=cv.GaussianBlur(og,(17,17),7)
      og=cv.dilate(og,dil,iterations=3)
      og=cv.adaptiveThreshold(og,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,7,4)
      og= cv.Canny(og,150,200)
      lines = cv.HoughLines(og,1,np.pi/180,50,None,0,0)
      #print(len(lines))
      if lines is None:
          x, img_data = cv2.imencode('.jpg', og)
          raw_bytes = img_data.tobytes()
          yield (b'--frame\r\n'
          b'Content-Type: image/jpeg\r\n\r\n' + raw_bytes +b'\r\n')
          continue
      og=frame
      parallel=[]
      thresh=20
      athresh=10
      for i in lines:
           rho,theta = i[0]
           for  l in lines:
               if (l[0][0]-rho>thresh) and l[0][1]-theta<(athresh*np.pi/180):
                   parallel.append([(rho,l[0][0]),theta])
      rho=0
      rhol=0
      rhor=0
      theta=0
      #cv.line(og,(0,0),(640,480),(255,0,0),7)
      if len(parallel)!=0:
          for par in parallel:
              rho1,rho2 = par[0]
              rho += (rho1+rho2)/2
              theta+=par[1]
          theta*=1/len(parallel)
          rho*=1/len(parallel)
          a = np.cos(theta) 
          b = np.sin(theta)
          x0 = a*rho
          y0 = b*rho
          x1 = int(x0 + 1000*(-b))
          y1 = int(y0 + 1000*(a))
          x2 = int(x0 - 1000*(-b))
          y2 = int(y0 - 1000*(a))
          cv.line(og,(x1,y1),(x2,y2),(255,0,0),5)
          for par in parallel:
              rho1,rho2=par[0]
              rhol+=max(par[0])
              rhol+=min(par[0])
          rhol*=1/len(parallel)
          rhor*=1/len(parallel)
          a = np.cos(theta)
          b = np.sin(theta)
          x0 = a*rhol
          y0 = b*rhol
          x1 = int(x0 + 1000*(-b))
          y1 = int(y0 + 1000*(a))
          x2 = int(x0 - 1000*(-b))
          y2 = int(y0 - 1000*(a))
          cv.line(og,(x1,y1),(x2,y2),(0,255,0),5)
          a = np.cos(theta)
          b = np.sin(theta)
          x0 = a*rhor
          y0 = b*rhor
          x1 = int(x0 + 1000*(-b))
          y1 = int(y0 + 1000*(a))
          x2 = int(x0 - 1000*(-b))
          y2 = int(y0 - 1000*(a))
          cv.line(og,(x1,y1),(x2,y2),(0,0,255),5)
      '''
      gray=cv.cvtColor(og,cv.COLOR_BGR2GRAY)
      gray = cv.GaussianBlur(gray,(5,5), 1)
      
      ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
      contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      #https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/
      clist=[]
      for c in contours:
          for eps in np.linspace(0.001, 0.05, 10):
	# approximate the contour
              peri = cv2.arcLength(c, True)
              approx = cv2.approxPolyDP(c, eps * peri, False)
	# draw the approximated contour on the image
              output = og.copy()
              #cv2.drawContours(og, [approx], -1, (0, 255, 0), 3)
              text = "eps={:.4f}, num_pts={}".format(eps, len(approx)) 
              if len(approx)==15:
                  clist.append(approx)
          cv.drawContours(og, [approx],-1,(0,244,144),5)
          #clist.append(approx)
      if clist!=[]:
          avg=sum(clist)/len(clist)
          avg=avg.reshape(-1,1,2)
          avg= avg.astype(np.int32)
      #print(avg[:,0,:])
          cv.drawContours(og,[avg],-1,(125,234,34),7)
      '''
      time.sleep(0.001)
#      break
      #raw2=cv.addWeighted(frame,0.7,og,0.3,0)
      raw2=og
      x, img_data = cv2.imencode('.jpg', raw2)
      raw_bytes = img_data.tobytes()
     # l.release()
      yield (b'--frame\r\n'
      b'Content-Type: image/jpeg\r\n\r\n' + raw_bytes + b'\r\n')

@app.route('/video')
def video():
    return  Response(gen_vid(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/proc_vid')
def proc_vid():
    return Response(gen_proc_vid(), mimetype='multipart/x-mixed-replace; boundary=frame')
    #gen_proc_vid()
@app.route("/log")
def log():
	try: 
	    open = open("log.txt","x") 
	    open.close()
	except:pass
	write= open("log.txt","a")
	write.write(request.args.get("logdata"))
	write.close()
@app.route("/controls/<dir>",methods=['POST'])
def mover(dir):
	pi_client.execute_command(dir)
	return render_template('gui.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
