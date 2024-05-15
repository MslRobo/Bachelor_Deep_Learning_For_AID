import os
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

parser = argparse.ArgumentParser(
    description="Analyzing output statistics from the main software"
)
parser.add_argument("-s",
                    "--source",
                    help="Select the source directory, This is expecting a directory created from the session config file with its setup, expects only name of the directory",
                    type=str)

args = parser.parse_args()

def confMatrix(confMatrix, outputDir):
    print("\nConfusion matrix")
    print(confMatrix)
    for matrix in confMatrix:
        # print(matrix)
        matrixOutputDir = os.path.join(outputDir, matrix['statInfo']['detection'], matrix['statInfo']['tracking'], matrix['statInfo']['image_enhancement'])
        if not os.path.exists(matrixOutputDir):
            os.makedirs(matrixOutputDir)
        confMtx = np.array([
            [matrix['tp'], matrix['fn']],
            [matrix['fp'], matrix['tn']]
        ])
        confMtx = confMtx.astype(int)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confMtx, annot=True, fmt="d", cmap="Blues", cbar=False)
        title = f"Detection: {matrix['statInfo']['detection']}, Tracking: {matrix['statInfo']['tracking']}, Img_enh: {matrix['statInfo']['image_enhancement']}, Noise_type: {matrix['statInfo']['noise_type']}"
        plt.title(f"Confusion Matrix {title}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks([0.5, 1.5], ["Positive", "Negative"])
        plt.yticks([0.5, 1.5], ["Positive", "Negative"], rotation=0)
        print(matrix['statInfo']['file'])
        dataFile = matrix['statInfo']['file'].split('.')[0] if matrix['statInfo']['file'].endswith('.json') else matrix['statInfo']['file']
        filename = f"{matrixOutputDir}/Confusion_matrix_{dataFile}_{matrix['statInfo']['noise_type']}.png"
        plt.savefig(filename)
        plt.close()

def datasetConfMatrix(matrix, outputDir):
    filename = os.path.join(outputDir, "Confusion_matrix.png")
    confMtx = np.array([
        [matrix['tp'], matrix['fn']],
        [matrix['fp'], matrix['tn']]
    ])
    confMtx = confMtx.astype(int)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confMtx, annot=True, fmt="d", cmap="Blues", cbar=False)
    title = f"Detection: {matrix['statInfo']['detection']}, Tracking: {matrix['statInfo']['tracking']}, Img_enh: {matrix['statInfo']['image_enhancement']}, Noise_type: {matrix['statInfo']['noise_type']}"
    plt.title(f"Confusion Matrix {title}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ["Positive", "Negative"])
    plt.yticks([0.5, 1.5], ["Positive", "Negative"], rotation=0)
    print(matrix['statInfo']['file'])
    dataFile = matrix['statInfo']['file'].split('.')[0] if matrix['statInfo']['file'].endswith('.json') else matrix['statInfo']['file']
    plt.savefig(filename)
    plt.close()

def brightness_graphing(graphs, outputDir):

    for graph in graphs:
        if None in graph['brightness_level']:
            continue
        graphOutputDir = os.path.join(outputDir, graph['statInfo']['detection'], graph['statInfo']['tracking'], graph['statInfo']['image_enhancement'])
        
        if not os.path.exists(graphOutputDir):
            os.makedirs(graphOutputDir)
        
        for value in graph['y_value']:
            x_value = np.array(graph['brightness_level'])
            y_value = np.array(graph['y_value'][value])
            
            indices = np.argsort(x_value)
            sorted_x = x_value[indices]
            sorted_y = y_value[indices]

            # print(f"\n Filename: {graph['file']}, folder: {graphOutputDir}")
            # print(f"Brightness: Value: {sorted_x}, {value}: {sorted_y} \n")
            plt.figure(figsize=(8,4))
            plt.plot(sorted_x, sorted_y)
            title = f"Detection: {graph['statInfo']['detection']}, Tracking: {graph['statInfo']['tracking']}, Img_enh: {graph['statInfo']['image_enhancement']}, Noise_type: {graph['statInfo']['noise_type']}"
            plt.title(title)
            plt.xlabel('Brightness Value')
            plt.ylabel(value)
            filename = f"{graphOutputDir}/{graph['statInfo']['noise_type']}_{value}.png"
            plt.savefig(filename)
            plt.close()

def over_time_performance(df, outputDir):
    filename = os.path.join(outputDir, f'over_time_performance.png')

    plt.figure(figsize=(12, 6))
    plt.plot(df['frame_number'], df['mean_detection_time'], label='Mean Detection Time')
    plt.plot(df['frame_number'], df['min_detection_time'], label='Min Detection Time')
    plt.plot(df['frame_number'], df['max_detection_time'], label='Max Detection Time')
    plt.xlabel('Frame number')
    plt.ylabel('Time (ms)')
    plt.title('Detection Time Over Time')
    plt.legend()
    plt.savefig(filename)
    plt.close()

#
def detection_accuracy_bar(df, outputDir):
    filename = os.path.join(outputDir, f'detection_accuracy_bar.png')
    print(df['detection_accuracy'])
    print(df['detection_accuracy_adjusted'])
    print(df['tracking_accuracy'])
    print(df['incident_accuracy'])
    accuracy_data = df[['detection_accuracy', 'detection_accuracy_adjusted', 'tracking_accuracy', 'incident_accuracy']]
    long_format = accuracy_data.melt(value_vars=['detection_accuracy', 'detection_accuracy_adjusted', 'tracking_accuracy', 'incident_accuracy'], var_name='Metric', value_name='Percentage')


    sns.barplot(x="Metric", y="Percentage", data=long_format)
    plt.xlabel('Metric')
    plt.ylabel('Percentage')
    plt.title('Detection and Tracking Accuracy Metrics')
    plt.savefig(filename)
    plt.close()

#
def tracking_analysis_bar(df, outputDir):
    filename = os.path.join(outputDir, f'tracking_analysis_bar.png')

    tracking_data = df[['tracking_id_switches', 'tracking_id_duplicates']]
    long_format = tracking_data.melt(value_vars=['tracking_id_switches', 'tracking_id_duplicates'], var_name='', value_name='Count')

    sns.barplot(x="", y="Count", data=long_format)
    plt.ylabel('Count')
    plt.savefig(filename)
    plt.close()

def incident_analysis_graph(df, outputDir):
    filename = os.path.join(outputDir, f'incident_analysis_graph.png')

    plt.plot(df['frame_number'], df['number_of_wrong_classes'], label='Number of wrong classes')
    plt.plot(df['frame_number'], df['false_positive_detections'], label='False positive detections')
    plt.xlabel('Frame number')
    plt.ylabel('Count')
    plt.title('Incident Reporting Over Time')
    plt.legend()
    plt.savefig(filename)
    plt.close()

# def queue_analysis(df, outputDir):
#     plt.figure(figsize=(12, 6))
#     plt.plot(df['frame_number'], df['avg_queue_length'], label='Average Queue Length')
#     plt.xlabel('Frame number')
#     plt.ylabel('Queue Length')
#     plt.title('Queue Length Over Time')
#     plt.legend()
#     plt.savefig(filename)
#     plt.close()

#
def detection_time_analysis(df, outputDir):
    filename = os.path.join(outputDir, f'detection_time_analysis.png')

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[['mean_detection_time', 'mean_tracking_time', 'mean_total_time']])
    plt.ylabel('Time (ms)')
    plt.title('Distribution of Detection, Tracking, and Total Times')
    filename = os.path.join(outputDir, f'')
    plt.savefig(filename)
    plt.close()

def detection_heatmap(df, outputDir):
    filename = os.path.join(outputDir, f'detection_heatmap.png')

    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=df['x_coords'], y=df['y_coords'], cmap="Reds", shade=True, bw_adjust=.5)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Heatmap of Detection Locations')
    plt.savefig(filename)
    plt.close()

#
# def roc_recall_curves(df, outputDir):
#     rocFilename = outputDir
#     recallFilename = os.path.join(outputDir, f'recall.png')

#     true_labels = df['true_labels']
#     pred_score = df['pred_scores']
#     print(f'True labels length: {len(true_labels)}')
#     print(f'pred labels length: {len(pred_score)}')

#     classes = np.unique(true_labels)
#     true_labels_bin = label_binarize(true_labels, classes=classes)

#     for i in range(len(classes)):
#         class_to_id = {0: 'None',1: 'car', 2: 'person', 3: 'truck', 4: 'bus', 5: 'bike', 6: 'motorbike', 10: 'Road anomaly'}
#         rocClassFilename = os.path.join(rocFilename, f"_{class_to_id[i]}_roc.png")
#         print(pred_score)
#         fpr, tpr, _ = roc_curve(true_labels_bin[:, i], pred_score[:, i])
#         roc_auc = auc(fpr, tpr)

#         plt.figure(figsize=(10, 5))
#         plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver Operating Characteristic')
#         plt.legend(loc="lower right")
#         plt.savefig(rocClassFilename)
#         plt.close()

#     precision, recall, _ = precision_recall_curve(true_labels, pred_score)
#     pr_auc = auc(recall, precision)


#     plt.figure(figsize=(10, 5))
#     plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall curve')
#     plt.legend(loc="lower left")
#     plt.savefig(recallFilename)
#     plt.close()



def system_load_analysis(df, outputDir):
    filename = os.path.join(outputDir, f'system_load_analysis.png')
    
    plt.figure(figsize=(14, 7))

    plt.subplot(2, 2, 1)
    plt.plot(df['frame_number'], df['gpu_load_percent'], label='GPU Load')
    plt.xlabel('Time')
    plt.ylabel('GPU Load (%)')
    plt.title('GPU Load Over Time')

    plt.subplot(2, 2, 2)
    plt.plot(df['frame_number'], df['gpu_memory_usage'], label='GPU Memory Usage')
    plt.xlabel('Time')
    plt.ylabel('Memory Usage (MB)')
    plt.title('GPU Memory Usage Over Time')

    plt.subplot(2, 2, 3)
    plt.plot(df['frame_number'], df['cpu_usage'], label='CPU Usage')
    plt.xlabel('Time')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage Over Time')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def main():
    if not args.source:
        print("No source file specified")
        return
    
    sourceBaseDir = r'.\\data\\output'
    baseOutputDir = r'.\\data\\graphics_output'
    sourceDir = os.path.join(sourceBaseDir, args.source)
    outputDir = os.path.join(baseOutputDir, args.source)

    i = 0
    try:
        while os.path.exists(outputDir):
            alt_path = f"{args.source}_{i}"
            outputDir = os.path.join(baseOutputDir, alt_path)
            i += 1

            if i == 10000:
                raise TimeoutError
    except TimeoutError as e:
        print(f"The directory part has timed out, i = {i}")
        return


    statisticList = []

    # Collecting data
    for entry in os.listdir(sourceDir):
        runDir = os.path.join(sourceDir, entry)
        
        for jsonFiles in os.listdir(runDir):
            # if jsonFiles.startswith("Video2"):
            #     continue
            if jsonFiles.split(".")[1] != "json":
                continue

            sourceFile = os.path.join(runDir, jsonFiles)

            try:
                with open(sourceFile, 'r') as file:
                    data = json.load(file)
                    statisticList.append({'file': file,'data': data, 'filename': jsonFiles})
            except FileNotFoundError:
                print(f"The file was not found. {sourceFile}")
            except json.JSONDecodeError:
                print("Error decoding JSON.")
            except Exception as e:
                print(f"An error occurred: {e}")

    
    # Processing data
    graphs = []
    graphDict = {'statInfo': {}}
    confusionMatrix = []
    dataframe = {}
    dataframes = []

    detection_accuracy = []
    detection_accuracy_adjusted = []
    tracking_accuracy = []
    incident_accuracy = []
    tracking_id_switches = []
    tracking_id_duplicates = []
    mean_detection_time = []
    mean_tracking_time = []
    mean_total_time = []
    true_labels = []
    pred_labels = []
    confMatrixValues = {
        'statInfo': None,
        'tn': 0,
        'fp': 0,
        'fn': 0,
        'tp': 0
    }
    for stat in statisticList:
        file = stat['file']
        filename = stat['filename']
        stat = stat['data']
        keys = ['brightness_level', 'mean_detection_time', 'mean_tracking_time', 'detection_accuracy', 'cars_detected_outside_mask', 'resolution']
        if not all(key in stat for key in keys):
            continue

        #Statistics cleanup
        # print(stat)
        statInfo = {'file': filename,'detection': stat['detection'], 'tracking': stat['tracking'], 'image_enhancement': stat['image_enhancement'], 'noise_type': stat['noise_type'], 'resolution': stat['resolution']}
        statInGraph = False

        if stat['brightness_level'] == 0 or stat['brightness_level'] == None:
            tn = stat['total_number_of_detections'] - stat['total_number_of_valid_detections']
            confusionMatrix.append({'statInfo': statInfo, 'tn': tn, 'fp': stat['false_positive_detections'], 'fn': stat['missed_detections'], 'tp': stat['total_number_of_valid_detections']})
            confMatrixValues['statInfo'] = statInfo
            confMatrixValues['tn'] += tn
            confMatrixValues['fp'] += stat['false_positive_detections']
            confMatrixValues['fn'] += stat['missed_detections']
            confMatrixValues['tp'] += stat['total_number_of_valid_detections']

        try:
            for graph in graphs:

                if graph.get('statInfo', None) == statInfo:
                    graph['file'].append(file)
                    graph['brightness_level'].append(stat['brightness_level'])
                    graph['y_value']['mean_detection_time'].append(stat['mean_detection_time'])
                    graph['y_value']['mean_tracking_time'].append(stat['mean_tracking_time'])
                    graph['y_value']['detection_accuracy'].append(stat['detection_accuracy'])
                    graph['y_value']['cars_detected_outside_mask'].append(stat['cars_detected_outside_mask'])
                    statInGraph = True
                    
            if not statInGraph:
                tempDict = {
                    'statInfo': statInfo,
                    'file': [file],
                    'brightness_level': [stat['brightness_level']],
                    'y_value': {
                        'mean_detection_time': [stat['mean_detection_time']],
                        'mean_tracking_time': [stat['mean_tracking_time']],
                        'detection_accuracy': [stat['detection_accuracy']],
                        'cars_detected_outside_mask': [stat['cars_detected_outside_mask']],
                    }
                }
                graphs.append(tempDict)
        except KeyError as err:
            print(f"Key error: {err}")
            continue
        

        detection_accuracy.append(stat['detection_accuracy'])
        detection_accuracy_adjusted.append(stat['detection_accuracy_adjusted'])
        tracking_accuracy.append(stat['tracking_accuracy'])
        incident_accuracy.append(stat['incident_accuracy'])
        tracking_id_switches.append(stat['tracking_id_switches'])
        tracking_id_duplicates.append(stat['tracking_id_duplicates'])
        mean_detection_time.append(stat['mean_detection_time'])
        mean_tracking_time.append(stat['mean_tracking_time'])
        mean_total_time.append(stat['mean_total_time'])
        video_data = {'filename': filename, 'img_enh': statInfo['image_enhancement'], 'frame_data': {}, 'centerpoints': {'x': [], 'y': []}}
        for frame in stat['frame_data']:
            
            if video_data['frame_data'] == {}:
                video_data['frame_data'] = {
                    'frame_number': [frame['frame_data']['frame_number']],
                    'current_time': [frame['frame_data']['current_time']],
                    'mean_detection_time': [frame['mean_detection_time']],
                    'min_detection_time': [frame['min_detection_time']],
                    'max_detection_time': [frame['max_detection_time']],
                    'number_of_wrong_classes': [frame['number_of_wrong_classes']],
                    'false_positive_detections': [frame['false_positive_detections']],
                    'gpu_load_percent': [frame['computational_data']['gpu_load_percent']],
                    'gpu_memory_usage': [frame['computational_data']['gpu_memory_usage']],
                    'cpu_usage': [frame['computational_data']['cpu_usage']],
                }
            else:
                video_data['frame_data']['frame_number'].append(frame['frame_data']['frame_number'])
                video_data['frame_data']['current_time'].append(frame['frame_data']['current_time'])
                video_data['frame_data']['mean_detection_time'].append(frame['mean_detection_time'])
                video_data['frame_data']['min_detection_time'].append(frame['min_detection_time'])
                video_data['frame_data']['max_detection_time'].append(frame['max_detection_time'])
                video_data['frame_data']['number_of_wrong_classes'].append(frame['number_of_wrong_classes'])
                video_data['frame_data']['false_positive_detections'].append(frame['false_positive_detections'])
                video_data['frame_data']['gpu_load_percent'].append(frame['computational_data']['gpu_load_percent'])
                video_data['frame_data']['gpu_memory_usage'].append(frame['computational_data']['gpu_memory_usage'])
                video_data['frame_data']['cpu_usage'].append(frame['computational_data']['cpu_usage'])

        video_data['centerpoints']['x'] = stat['detection_data']['centerpoint']['x']
        video_data['centerpoints']['y'] = stat['detection_data']['centerpoint']['y']

        if statInfo['image_enhancement'] in dataframe:
            dataframe[statInfo['image_enhancement']]['video_stats'].append(video_data)
        else:
            dataframe[statInfo['image_enhancement']] = {
                'video_stats': [video_data],
                'dataset_stats': {}
            }
        
        true_labels = true_labels + stat['detection_data']['true_labels']
        pred_labels = pred_labels + stat['detection_data']['predicted_labels']

        dataframe[statInfo['image_enhancement']]['dataset_stats'] = {
            'img_enh': statInfo['image_enhancement'],
            'detection_accuracy': sum(detection_accuracy) / len(detection_accuracy),
            'detection_accuracy_adjusted': sum(detection_accuracy_adjusted) / len(detection_accuracy_adjusted),
            'tracking_accuracy': sum(tracking_accuracy) / len(tracking_accuracy),
            'tracking_id_switches': sum(tracking_id_switches) / len(tracking_id_switches),
            'tracking_id_duplicates': sum(tracking_id_duplicates) / len(tracking_id_duplicates),
            'incident_accuracy': sum(incident_accuracy) / len(incident_accuracy),
            'mean_detection_time': sum(mean_detection_time) / len(mean_detection_time),
            'mean_tracking_time': sum(mean_tracking_time) / len(mean_tracking_time),
            'mean_total_time': sum(mean_total_time) / len(mean_total_time),
            'true_labels': true_labels,
            'pred_labels': pred_labels
        }
        


    videoOutputPath = os.path.join(outputDir, )

    # Visualizing data
    confMatrix(confusionMatrix, outputDir)

    baseBaseOutputDir = outputDir
    for key in dataframe:
        dafa = dataframe[key]
        baseOutputDir = os.path.join(baseBaseOutputDir, dafa['dataset_stats']['img_enh'])
        print("\n")
        print(dafa['dataset_stats']['img_enh'])
        if not os.path.exists(baseOutputDir):
            os.mkdir(baseOutputDir)
        for video in dafa['video_stats']:
            filename = video['filename']
            outputDir = os.path.join(baseOutputDir, filename.split('.')[0])
            if not os.path.exists(outputDir):
                os.mkdir(outputDir)
            df = pd.DataFrame(video['frame_data'])
            
            heatmap_vals = {
                'x_coords': video['centerpoints']['x'],
                'y_coords': video['centerpoints']['y']
            }

            # heatmap_vals = pd.DataFrame(heatmap_vals)

            over_time_performance(df, outputDir)
            incident_analysis_graph(df, outputDir)
            detection_heatmap(heatmap_vals, outputDir)
            system_load_analysis(df, outputDir)
        
        roc_values = {
            'true_labels': dafa['dataset_stats']['true_labels'],
            'pred_scores': dafa['dataset_stats']['pred_labels']
        }
        outerDf = pd.DataFrame(dafa['dataset_stats'])
        datasetConfMatrix(confMatrixValues, baseOutputDir)
        detection_accuracy_bar(outerDf, baseOutputDir)
        tracking_analysis_bar(outerDf, baseOutputDir)
        detection_time_analysis(outerDf, baseOutputDir)
        # roc_recall_curves(roc_values, baseOutputDir)



    # print("Hey :)")

if __name__ == '__main__':
    main()
