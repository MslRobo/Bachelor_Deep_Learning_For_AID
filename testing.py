import argparse
import json
import os
import torchaudio
import keras

parser = argparse.ArgumentParser(
    description="Tester"
)
parser.add_argument("-s",
                    "--source",
                    help="Testing testing",
                    type=str)


args = parser.parse_args()

def main():
    with open(os.path.join(r'.\\data\\incidents\\Video1', r'annotations.json')) as f:
        data = json.load(f)
    counter = 0
    ids = [3, 4, 5, 6]
    for line in data:
        if counter == 0:
            counter += 1
            continue
        for i in data[line]['instances']:
            if i['type'] != 'bbox':
                print(i)
        #     for attribute in i['attributes']:
        #         if attribute['id'] == 1:
        #             if attribute['id'] != 1:
        #                 print(attribute)
        #     # print(i['classId'])
        #     if i['classId'] in ids:
        #         # print(i)
        #         ids.remove(i['classId'])
        # if len(ids) == 0:
        #     break
        # if data[line]['instances']
        # print(f"Line: {counter}")
        # print(line)
        # print(data[line])
        # print("\n")
        # print("\n")
        # for i in data[line]['instances']:
        #     print(str(i) + '\n')
        # print("\n")
        # print("============================================================")
        # if counter == 10:
        #     break
        # counter += 1

if __name__ == '__main__':
    print(torchaudio.__version__)
    print(keras.__version__)
    # main()

'''
ClassId 6 = motorbike
ClassId 5 = bike
ClassId 4 = bus
ClassId 3 = truck
ClassId 2 = person
ClassId 1 = car

GroupId is the id of the attribute group for a class
Id is the id of the attribute in the attribute group
'''