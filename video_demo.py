from __future__ import print_function
import os
import time
from multiprocessing import Process, Queue, Value
import queue
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import argparse
from distutils.version import StrictVersion
from package import config as config, visualization_utils as vis_utils
import base64
from imutils.video import VideoStream
from datetime import datetime
from datetime import date
from sqldatabase import Image
 
smear = 2
threshold = smear*smear
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()
 
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*')
 
 
def load_model(inference_model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph
 
 
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
 
# Definite input and output Tensors for detection_graph
# Each box represents a part of the image where a particular object was detected
# Each score represent how level of confidence for each of the objects
# Score is shown on the result image, together with the class label
def run_inference_for_single_image(image, sess, tensor_dict):
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0') 
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
 
    return output_dict
 
def is_person(head_boxes, vest_boxes, person_box):
    head_flag = False
    vest_flag = False
    head_intersection_ratio = 0.6
    vest_intersection_ratio = 0.6
 
    #for head_box in head_boxes:
    #    head_flag = is_wearing_head(person_box, head_box, head_intersection_ratio)
    #    if head_flag:
    #        break
 
    #for vest_box in vest_boxes:
    #    vest_flag = is_wearing_vest(person_box, vest_box, vest_intersection_ratio)
    #    if vest_flag:
    #        break
 
    return head_flag, vest_flag
 
 
def post_message_process(run_flag, message_queue):
     
    while run_flag.value:
        try:
            camera_id, output_dict, image, min_score_thresh = message_queue.get(block=True, timeout=5)
            post_message(camera_id, output_dict, image, min_score_thresh=0.7)
        except queue.Empty:
            continue
 
 
def post_message(camera_id, output_dict, image, min_score_thresh):
    message = dict()
    message["timestamp"] = int(time.time() * 1000)
    message["cameraId"] = camera_id
 
    image_info = {}
    image_info["height"] = image.shape[0]
    image_info["width"] = image.shape[1]
    image_info["format"] = "jpeg"
 
    success, encoded_image = cv2.imencode('.jpg', image)
    content = encoded_image.tobytes()
    image_info["raw"] = base64.b64encode(content).decode('utf-8')
 
    message["image"] = image_info
 
    detection_scores = np.where(output_dict["detection_scores"] > min_score_thresh, True, False)
 
    detection_boxes = output_dict["detection_boxes"][detection_scores]
    detection_classes = output_dict["detection_classes"][detection_scores]
 
    head_boxes = detection_boxes[np.where(detection_classes == 1)]
    vest_boxes = detection_boxes[np.where(detection_classes == 2)]
    person_boxes = detection_boxes[np.where(detection_classes == 3)]
 
    persons = []
    for person_box in person_boxes:
        person = dict()
        person["head"], person["vest"] = is_person(head_boxes, vest_boxes, person_box)
        persons.append(person)
 
    message["persons"] = persons
    
    if len(persons) == 0:
        return False
 
    # try:
    #     headers = {'Content-type': 'application/json'}
    #     if len(persons):
    #         result = requests.post(config.detection_api, json=message, headers=headers)
    #         print(result)
    #         return True
    # except requests.exceptions.ConnectionError:
    #     print("Connect to backend failed")
    return False
 
 
def image_processing(graph, category_index, image_file_name, show_video_window):
 
    img = cv2.imread(image_file_name)
    image_expanded = np.expand_dims(img, axis=0)
 
    with graph.as_default():
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        with tf.Session() as sess:
            output_dict = run_inference_for_single_image(image_expanded, sess, tensor_dict)
 
            vis_utils.visualize_boxes_and_labels_on_image_array(
                img,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=4)
 
            if show_video_window:
                cv2.imshow('ppe', img)
                cv2.waitKey(5000)
## [create]
#create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()
## [create]
 
def video_processing(graph, category_index, video_file_name, show_video_window, camera_id, run_flag, message_queue): 
    person_count = 0
    frameWithPerson = None
    personFindJitter = 2        # Number of frames with no person before the person is assumed to have left
    framesNoPerson = personFindJitter
    # Open camera, or video file
    if camera_id is None:
        cap = cv2.VideoCapture(0) 
        #cap = cv2.VideoCapture(video_file_name) 
        output_fps = cap.get(cv2.CAP_PROP_FPS)
    else :
        print("[INFO] starting cameras...")
        cap = cv2.VideoCapture(int(camera_id))
        output_fps = 30
    # Open output video file
    video_output = 'output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, output_fps, (config.display_window_width, config.display_window_height))
    # Optionally open dislay window
    if show_video_window:
        cv2.namedWindow('ppe', cv2.WINDOW_NORMAL)
        if config.display_full_screen:
            cv2.setWindowProperty('ppe', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty('ppe', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    # For file read, set frame size based on arguments if specified size is supported
    if camera_id is None :
        if (config.capture_image_width, config.capture_image_height) in config.supported_video_resolution:
            print("video_processing:", "supported video resoulution")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_image_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_image_height)
 
#    video_output = "output.mp4"
    with graph.as_default():
        print("video_processing:", "default tensorflow graph")
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        with tf.Session() as sess:
            print("video_processing:", "tensorflow session")
            send_message_time = time.time()
            frame_counter = 0
            i = 0  # default is 0
            dbImage = Image("newdata.db")
            while True:
            # Get the next frame
                frame_counter += 1
                ret, frame = cap.read()
                if not ret :
                    break
                if frame is None:
                    print("video_processing:", "null frame")
                    break
                # Look for various objects in frame
                resized_frame = cv2.resize(frame, dsize=(640, 360)) # Fixed frame size
                image_expanded = np.expand_dims(resized_frame, axis=0)
                output_dict = run_inference_for_single_image(image_expanded, sess, tensor_dict)
 
                detection_scores = np.where(output_dict["detection_scores"] > 0.7, True, False)
                detection_boxes = output_dict["detection_boxes"][detection_scores]
                detection_classes = output_dict["detection_classes"][detection_scores]
 
                head_boxes = detection_boxes[np.where(detection_classes == 1)]
                vest_boxes = detection_boxes[np.where(detection_classes == 2)]
                person_boxes = detection_boxes[np.where(detection_classes == 3)]
                persons = []
                for person_box in person_boxes:
                    person = dict()
                    person["head"], person["vest"] = is_person(head_boxes, vest_boxes,
                                                                                person_box)
                    persons.append(person)
 
                vis_utils.visualize_boxes_and_labels_on_image_array(
                    frame,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=4)
 
                # Periodically que a message (not sure what that does)
                if time.time() - send_message_time > config.message_send_interval / 1000.0:
                    resized_frame = cv2.resize(frame,
                                                dsize=(config.storage_image_width, config.storage_image_height))
                    try:
                        message_queue.put_nowait(
                            (camera_id, output_dict, resized_frame, config.object_confidence_threshold))
                    except queue.Full:
                        print("message queue is full")
                    else:
                        send_message_time = time.time()
                # Show frame
                if show_video_window:
                    resized_frame = cv2.resize(frame,
                                                dsize=(config.display_window_width, config.display_window_height))
                    # Add text to frame to indicate what was found
                    height, width = resized_frame.shape[:2]
                    hat_count = 0
                    vest_count = 0
                    hat_and_vest_count = 0
                    for person in persons:
                        if person['head'] and person['vest']:
                            hat_and_vest_count += 1
                        elif person['head']:
                            hat_count += 1
                        elif person['vest']:
                            vest_count += 1
 
                    resized_frame = cv2.putText(resized_frame, "No of person: " + str(len(person_boxes)),
                                                (30, height - 170), cv2.FONT_HERSHEY_TRIPLEX, 1, (150, 100, 50), 2,
                                                cv2.LINE_AA)

                    #update the background model
                    fgMask = backSub.apply(frame)
                    # Transfer mask to numpy
                    (im_height, im_width) = frame.shape[:2]
     
                    # Clean up mask
                    iMask = np.array(fgMask.reshape((im_height, im_width)).astype(np.uint8))
                    iMask = (iMask>127).astype(np.uint8)
                    # Eliminate isolated pixels
                    oMask = iMask
                    for k in range(1,smear) :
                        oMask[:,:im_width-k-1] += iMask[:,k:im_width-1]
                        oMask[:,k:im_width-1] += iMask[:,:im_width-k-1]
                    iMask = oMask
                    for k in range(1,smear) :
                        oMask[:im_height-k-1,:] += iMask[k:im_height-1,:]
                        oMask[k:im_height-1,:] += iMask[:im_height-k-1,:]
                    oMask = oMask>threshold
                    iMask = oMask
                    for k in range(smear) :
                        oMask[:,:im_width-k-1] |= iMask[:,k:im_width-1]
                        oMask[:,k:im_height-1] |= iMask[:,:im_height-k-1]
                    iMask = oMask
                    for k in range(smear) :
                        oMask[:im_height-k-1,:] |= iMask[k:im_height-1,:]
                        oMask[k:im_height-1,:] |= iMask[:im_height-k-1,:]
                    oMask = 255 * oMask.astype(np.uint8)
                    # Show annotated frame
                    cv2.imshow('ppe', resized_frame)
                    cv2.imshow('FG Mask', fgMask)
                    cv2.imshow('Contours', oMask)
                    now = datetime.now()
                    #start_time = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    o_frame = cv2.bitwise_and(frame,frame,mask = oMask)
                    cv2.imshow('Masked', o_frame)
                    # Save image on first and last occurance of a person
                    pic_name = None
                    if len(person_boxes) > 0 :
                        if framesNoPerson >= personFindJitter :
                            print ("Person entered at "+ current_time)
                            pic_name = "firstframe" + str(frame_counter) + "_" + str(person_count)+ ".jpg"
                            person_count+=1
                        frameWithPerson=resized_frame
                        framesNoPerson = 0
                    else :
                        framesNoPerson += 1
                        if framesNoPerson == personFindJitter :
                            print ("Person left at "+ current_time)
                            pic_name = "lastframe" + str(frame_counter) + str(person_count)+ ".jpg"
                            person_count+=1
                    if pic_name is not None and frameWithPerson is not None :
                        cv2.imwrite('./Pictures/'+str(i)+'.jpg', frameWithPerson)
                        cv2.imwrite("./Pictures/" + pic_name , frameWithPerson)
                        cv2.imwrite('./images/'+str(i)+'.jpg', o_frame)
                        cv2.imwrite("./images/" + pic_name , o_frame)
                        with open("./Pictures/" + pic_name, 'rb') as f:
                            dbImage.create_database(name=pic_name, starttime=now, endtime=current_time, image=f.read())
                 
                    out.write(resized_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        run_flag.value = 0
                        break
                        
 
                # k = cv2.waitKey(30) & 0xff
                # if k == 27:
                #     break
 
    print("video_processing:", "releasing video capture")
    out.release()
    cap.release()
    cv2.destroyAllWindows()
 
def main():
    parser = argparse.ArgumentParser(description="head and Vest Detection", add_help=True)
    parser.add_argument("--model_dir", type=str, required=False,default="./model", help="path to model directory")
    parser.add_argument("--video_file_name", type=str, required=False,default="input.mp4", help="path to video file, or camera device, i.e /dev/video1")
    parser.add_argument("--show_video_window", type=int, required=False,default=1, help="the flag for showing the video window, 0 is not dispaly, 1 display")
    parser.add_argument("--camera_id", type=str, required=False,default=None, help="camera identifier")
    args = parser.parse_args()
 
    frozen_model_path = os.path.join(args.model_dir, "frozen_inference_graph.pb")
    if not os.path.exists(frozen_model_path):
        print("frozen_inference_graph.db file is not exist in model directory")
        exit(-1)
    print("loading model")
    graph = load_model(frozen_model_path)
    category_index = {1: {'id': 1 , 'name': 'head'},
                      2: {'id': 2, 'name': 'vest'},
                      3: {'id': 3, 'name': 'person'}}
     
    print("start message queue")
    run_flag = Value('i', 1)
    message_queue = Queue(1)
    p = Process(target=post_message_process, args=(run_flag, message_queue))
    p.start()
    print("video processing")
    video_processing(graph, category_index, './data/input/' + args.video_file_name, args.show_video_window, args.camera_id, run_flag, message_queue)
    print(message_queue.get())
    p.join()
     
    #image_processing(graph, category_index, './examples/002.jpg', True)
 
if __name__ == '__main__':
    main()
