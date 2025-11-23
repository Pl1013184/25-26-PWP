from Flask import Flask, Response
import cv2

app = Flask(__name__)
video = cv2.VideoCapture(0)

def gen_vid():
    while True:
        tester, frame = video.read()
        if not tester:
            continue
        raw = frame
        # sets up for processing
        x, img_data = cv2.imencode('.jpg', raw)
        raw_bytes = img_data.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + raw_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_vid(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
