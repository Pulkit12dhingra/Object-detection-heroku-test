# Required Imports
from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import io
from io import StringIO
from PIL import Image
from io import BytesIO
import base64
# requred imports 
import gluoncv as gcv
from gluoncv.utils import try_import_cv2
import mxnet as mx
import numpy as np
from engineio.payload import Payload

# initialization
Payload.max_decode_packets = 40
async_mode = None

# start flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

# Load the model
net = gcv.model_zoo.get_model('yolo3_mobilenet1.0_voc', pretrained=True)
net.reset_class(classes=['person'], reuse_weights=['person'])

mydict={}
for i,name in enumerate(net.classes):
    mydict[i]=name


def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        socketio.emit('my_response',
                      {'data': 'Analysing....', 'count': count})


@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)


@socketio.event
def connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('my_response', {'data': 'Connected', 'count': 0})


@socketio.event
def disconnect_request():
    @copy_current_request_context
    def can_disconnect():
        disconnect()

@socketio.event
def image(data_image):
    session['receive_count'] = session.get('receive_count', 0) + 1
    sbuf = StringIO()
    sbuf.write(data_image)
    try:
        image =np.array(Image.open(BytesIO(base64.b64decode(data_image))))
        frame = mx.nd.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype('uint8')
        rgb_nd, frame = gcv.data.transforms.presets.yolo.transform_test(frame, short=512, max_size=700)
        class_IDs, scores, bounding_boxes = net(rgb_nd)
        labels=class_IDs[0]
        labels = labels.asnumpy()
        cls_id = int(labels.flat[0]) if labels is not None else -1

        # Display the result
        if cls_id!=-1:
            msg="There is a "+mydict[cls_id]+ " in front"
        else:
            msg='No object detected'
        emit('my_response',
             {'data':msg,"count":session['receive_count']})
    except:
        pass



if __name__ == '__main__':
    socketio.run(app)
