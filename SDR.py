import warnings
import os
from hmmlearn import hmm
import numpy as np
from librosa.feature import mfcc
import librosa
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from tabulate import tabulate

def buildDataSet(dir,rte):
    # Filter out the wav audio files under the dir
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    train_dataset = {}
    test_dataset = {}
    train = random.sample(fileList, 456)
    test= random.sample(fileList, 196)
    cnt=1
    # Calculate percent of each train and test
    nm = int(rte*50)
    rnd = random.sample(range(0,100), nm)
    testingDataCount = 0
    trainDataCount = 0

    for fileName in test:
        label = fileName.split('-')[0]
        feature = extract_mfcc(dir+fileName).T
        if label not in test_dataset.keys():
            test_dataset[label] = []
            test_dataset[label].append(feature)
            testingDataCount += 1
        else:
            exist_feature = test_dataset[label]
            exist_feature.append(feature)
            test_dataset[label] = exist_feature
            testingDataCount += 1

    for fileName in train:
        label = fileName.split('-')[0]
        feature = extract_mfcc(dir+fileName).T
        if label not in train_dataset.keys():
            train_dataset[label] = []
            train_dataset[label].append(feature)
            trainDataCount += 1
        else:
            exist_feature = train_dataset[label]
            exist_feature.append(feature)
            train_dataset[label] = exist_feature
            trainDataCount += 1
        

    # for fileName in fileList:
    #     label = fileName.split('-')[0]
    #     # print(label)
    #     feature = extract_mfcc(dir+fileName).T
    #     if cnt in rnd:
    #         if label not in train_dataset.keys():
    #             train_dataset[label] = []
    #             train_dataset[label].append(feature)
    #             trainDataCount += 1
    #         else:
    #             exist_feature = train_dataset[label]
    #             exist_feature.append(feature)
    #             train_dataset[label] = exist_feature
    #             trainDataCount += 1
    #     else:
    #         if label not in train_dataset.keys():
    #             train_dataset[label] = []
    #             train_dataset[label].append(feature)
    #             trainDataCount += 1
    #         else:
    #             exist_feature = train_dataset[label]
    #             exist_feature.append(feature)
    #             train_dataset[label] = exist_feature
    #             trainDataCount += 1
    #     if (cnt == 100):
    #         cnt = 1
    #         rnd = random.sample(range(0, 100), 12)
    #     else:
    #         cnt += 1

    print(trainDataCount, "TRAIN DATA")
    print(testingDataCount, "TEST DATA","/n")
    print(trainDataCount + testingDataCount, "==== Total Dataset Build ====")

    return train_dataset,test_dataset


def extract_mfcc(full_audio_path):
    wave, sample_rate =  librosa.load(full_audio_path)
    mfcc_features = mfcc(wave, sample_rate )
    return mfcc_features

### Gussian HMM
def train_HMM(dataset):
    Models = {}
    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=4)
        trainData = dataset[label]
        trData = np.vstack(trainData)
        model.fit(trData)
        Models[label] = model
    return Models

### Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    ### ignore warning message of readfile
    warnings.filterwarnings('ignore')

    ### Step.1 Loading data
    trainDir = 'car_sound_dataset2/'
    print('Step.1 data loading...')
    trainDataSet,testDataSet = buildDataSet(trainDir,rte=0.30)
    print("Finish prepare the data")


    ### Step.2 Training
    print('Step.2 Training model...')
    hmmModels = train_HMM(trainDataSet)
    print("Finish training of the GMM_HMM models")

    ### Step.3 predict test data
    acc_count = 0
    all_data_count = 0
    real = []
    pred = []
    result = []
    for label in testDataSet.keys():
        feature = testDataSet[label]
        for index in range(len(feature)):
            all_data_count+=1
            scoreList = {}
            labelScore = {}

            normal = 0
            altenator = 0
            bearing = 0
            kompresor = 0
            waterpump = 0
            idler = 0

            normalTrue = 0
            altenatorTrue = 0
            bearingTrue = 0
            kompresorTrue = 0
            waterpumpTrue = 0
            idlerTrue = 0

            for model_label in hmmModels.keys():
                model = hmmModels[model_label]
                score = model.score(feature[index])
                scoreList[model_label] = score
            predict = max(scoreList, key=scoreList.get)
            real.append(label)
            pred.append(predict)
            scoreFormat = np.exp(score)
            temp = [all_data_count, label, predict, scoreFormat]
            result.append(temp)
            print(label, "has predicted result =>", predict)
            if predict == label:
                acc_count+=1

            if label == 'normal':
                normal+=1
                if predict == label:
                    normalTrue+=1
                    labelScore[model_label] = round(((normal/normalTrue)*100.0),3)
            
            if label == 'altenator':
                altenator+=1
                if predict == label:
                    altenatorTrue+=1
                    labelScore[model_label] = round(((altenator/altenatorTrue)*100.0),3)
            
            if label == 'bearing':
                bearing+=1
                if predict == label:
                    bearingTrue+=1
                    labelScore[model_label] = round(((bearing/bearingTrue)*100.0),3)

            if label == 'kompresor':
                kompresor+=1
                if predict == label:
                    kompresorTrue+=1
                    labelScore[model_label] = round(((kompresor/kompresorTrue)*100.0),3)

            if label == 'idler':
                idler+=1
                if predict == label:
                    idlerTrue+=1
                    labelScore[model_label] = round(((idler/idlerTrue)*100.0),3)

            if label == 'waterpump':
                waterpump+=1
                if predict == label:
                    waterpumpTrue+=1
                    labelScore[model_label] = round(((waterpump/waterpumpTrue)*100.0),3)

    print(all_data_count,'===== ini all data yang dilakukan tesing ======')
    print(all_data_count,'==== Data Tested ====')
    print(acc_count,'>>>>> data has success predicted <<<<')

    accuracy = round(((acc_count/all_data_count)*100.0),3)
    # print(real,'ini real')
    # print(pred,'ini inipredict')
    cm = confusion_matrix(real, pred)
    np.set_printoptions(precision=2)
    classes = ["normal","altenator", "bearing", "idler", "kompresor", "waterpump"]
    plt.figure()
    plot_confusion_matrix(cm, classes=classes, normalize=True,
    title='Normalized confusion matrix')

    print("\n##########################################################################")
    print("######################## A-C-C-U-R-A-C-Y #################################")
    print("########################    ",accuracy,"%","   #################################")
    print("##########################################################################")

    print(tabulate(result, headers=["iteration", "Label", "Real", "Predict", "Score"], tablefmt='fancy_grid'))
    plt.show()

if __name__ == '__main__':
    main()


# def buildDataSet(dir,rte):
#     # Filter out the wav audio files under the dir
#     fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
#     train_dataset = {}
#     test_dataset = {}
#     cnt=1
#     # Calculate percent of each train and test
#     nm = int(rte*50)
#     rnd = random.sample(range(0,100), nm)
#     # st.write("Total dataset", len(fileList))

#     chart_data = pd.DataFrame(
#         np.random.randn(20, 6),
#         columns=["normal","altenator", "bearing", "idler", "kompresor", "waterpump"])
#     st.area_chart(chart_data)


#     for fileName in fileList:
#         label = fileName.split('-')[0]
#         print(label)
#         feature = extract_mfcc(dir+fileName).T
#         if cnt in rnd:
#             if label not in test_dataset.keys():
#                 test_dataset[label] = []
#                 test_dataset[label].append(feature)
#             else:
#                 exist_feature = test_dataset[label]
#                 exist_feature.append(feature)
#                 test_dataset[label] = exist_feature
#         else:
#             if label not in train_dataset.keys():
#                 train_dataset[label] = []
#                 train_dataset[label].append(feature)
#             else:
#                 exist_feature = train_dataset[label]
#                 exist_feature.append(feature)
#                 train_dataset[label] = exist_feature
#         if (cnt == 100):
#             cnt = 1
#             rnd = random.sample(range(0, 100), 12)
#         else:
#             cnt += 1
#     return train_dataset,test_dataset