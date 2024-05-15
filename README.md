# Bachelor Deep Learning For AID
This bachelor thesis is a deeper look into how we can use deep learning for automatic incident detection in road tunnels. Some of the drive behind this thesis is Norways "nullvisjonen" which is a project to have zero deaths or severly injured cases in traffic

# Eval:
To run evaluations on the output data run the following command from the root directory:
    py StatisticAnalyser.py -s *name of any directory inside the output directory*

# Test models on dataset:
To test the models on the dataset run the recommended command:

    python liveTrack.py -s StandardAnalysis.json --file ArbitraryValue --datamode json --filetype jpg --show 1

**Description of flags**:
- -m: Specify which object detection model to use. 
  - **Options**: yolov5, yolov5_trained, yolov7, yolov8
  - **Default**: yolov5
- -i: Specify which image enhancement method to use:
  - **Options**: none, gray_linear, gray_nonlinear, he, retinex_ssr, retinex_msr, mask
  - **Default**: none
- -f: Specify a filename to save the results to a file.
- -s: Specify how many frames should be skipped.
- -p: If given, a pre-trained model will be used. The model name must correspond to the folder name in the "pre-trained-models" directory.
- --datamode: Specify whether or not to expect a run config or not
- --filetype: Specify whether or not to expect an image dataset or a mp4 file
  - **Options**: mp4, jpg
- --show: Specify whether or not to show a live feed of the model running over the dataset.
- --downscale: Specify whether or not to run downscaling.
- *It is recommended to run the full command though a short verison of without flags could work.*


# Refrences:
This bachelor thesis expands opon the work conducted by Alexander Vedvik in his bachlor thesis. As this bachelor thesis was to improve opon his work alot of the source code will be the same and therefore proper separation of his original work and our work proved difficult as alot of the improvements had to be conducted inside his files. The bachelor thesis of Aleksander Vedvik can be found [here](https://hdl.handle.net/11250/3003555) and the corresponding github repository can be found [here](https://github.com/aleksander-vedvik/Bachelor/tree/main)
- Reused source code:
  - [TFODCourse](https://github.com/nicknochnack/TFODCourse), Nicholas Renotte
  - [Retinex Image Enhancement](https://github.com/aravindskrishnan/Retinex-Image-Enhancement), Aravind S Krishnan
  - [The AI Guy Code](https://github.com/theAIGuysCode/yolov4-deepsort), The AI Guy
  - [Deep SORT](https://github.com/nwojke/deep_sort), Nicolai Wojke

- Datasets:
  - [Master thesis](https://github.com/BerntA/CVEET), Bernt Andreas Eide
  - [YouTube video used as test dataset](https://www.youtube.com/watch?v=IOxxEJpXZGU&ab_channel=RedDFilm)
