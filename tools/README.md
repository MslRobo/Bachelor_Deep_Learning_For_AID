# Tools

To make the AID system easier to read, each task have been divided into separate files. The performance evaluator model handles image enhancement, detection, tracking and the performance part of the AID system. Hence the pseudo-code for `run.py` can be written as:

    odm = Detection_Model()
    otm = Tracking_Model()
    iem = Incident_Evaluator_Model()
    pem = Performance_Evaluator_Model(odm, otm, iem)

    for frame in video:
        frame = pem.image_enhancement(frame)
        pem.detect_and_track(frame)

        for track in pem.get_tracks():
            result = iem.evaluate(track)
            pem.evaluate(result)
            visualize(track)
        
        show(frame)

## Detection
The detection model is stored as an object in the file `detection_model.py`. The object handles which type of model is used and the execution of the model. The execution function is named **detect()** and it returns the model detections.

## Tracking
The tracking model is stored as an object in the file `tracking_model.py`. The object handles which type of model is used and the execution of the model. The execution function is named **track()** and it updates the tracks. The tracks are stored in the tracking model object and accessed via the performance evaluator method **pem.get_tracks()**

## Incident evaluator
The incident evaluator model is stored as an object in the file `incident_evaluator.py`. The object handles the evaluation of incidents, and returns the incident class and direction of each vehicle.

## Performance evaluator
The performance evaluator model is stored as an object in the file `performance_evaluator.py`. The performance object works as a collection for the image enhancement, detection and tracking parts of the AID system. The performance object measures the performance of each algorithm and prints the results to the console. It also returns the results in a string which can be used to write a file.

## Visualizer
The visualizer functions are stored in the file `visualize_objects.py`. These functions handles the visualization of bounding boxes and text on the frames.
