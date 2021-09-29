# requred imports 
import gluoncv as gcv
from gluoncv.utils import try_import_cv2
cv2 = try_import_cv2()
import mxnet as mx
import numpy as np
from collections import Counter


# Load the model
net = gcv.model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
net.hybridize()

cap = cv2.VideoCapture(0)

axes = None
NUM_FRAMES = 1 # you can change this
for i in range(NUM_FRAMES):
    # Load frame from the camera
    ret, frame = cap.read()
    # Image pre-processing
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    rgb_nd, frame = gcv.data.transforms.presets.yolo.transform_test(frame, short=512, max_size=700)

    # Run frame through network
    class_IDs, scores, bounding_boxs = net(rgb_nd)
    # map class ID to classes
    mydict={}
    for i,name in enumerate(net.classes):
        mydict[i]=name

    labels=class_IDs[0]
    labels = labels.asnumpy()
    cls_id = int(labels.flat[0]) if labels is not None else -1

    # Display the result
    if cls_id!=-1:
        print(mydict[cls_id])
    else:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()