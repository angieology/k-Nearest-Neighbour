import numpy as np
import operator


#class train(object):
train_x = np.linspace(1.0,10.0,num=100)[:, np.newaxis]
train_y = np.sin(train_x) + 0.1 * np.power(train_x, 2) + 0.5 * np.random.randn(100, 1)




def euclideanDistance(instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return np.sqrt(distance)
    #practice euclidean distance formula
    #data1 = [2, 2, 2, 'a']
    #data2 = [4, 4, 4, 'b']
    #distance = euclideanDistance(x_eval[0], x[0], 64)
    #print 'Distance: ' + repr(distance)
    #print ('unseen test label:' , t_eval[0])
    #print ('actual label:' , t[0])




def getNeighbors(trainingSet, trainingLabelSet, testInstance, k):
        distances = []

        length = len(testInstance) - 1
        for x in range(len(trainingSet)):
            formatItem = []
            dist = euclideanDistance(testInstance, trainingSet[x], length)
            formatItem = (trainingSet[x], trainingLabelSet[x])#just formatting data and label into one array item
            distances.append((formatItem, dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])#just order by closest and get the K closest
        return neighbors


    #trainSet = [[2, 2, 2 ], [4, 4, 4]]
    #trainLabelSet = ['a','b']
    #testInstance = [5, 5, 5]
  #  k = 1
 #   neighbors = getNeighbors(trainSet, trainLabelSet, testInstance, 1)
  #  print(neighbors)

    #print("------------------")
def getResponse(neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1][0]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]


    #neighbors = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
    #response = getResponse(neighbors)
    #print(response)


def getAccuracy(testSet, testSetLabel, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSetLabel[x][0] == predictions[x]:
                correct += 1
        return (correct / float(len(testSet))) * 100.0


   # testSet = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
   # predictions = ['a', 'a', 'a']
   # accuracy = getAccuracy(testSet, predictions)
   # print(accuracy)
def formatData(dataSet, labelSet):
        doneSet = []
        for x in range(len(labelSet)):
            doneSet.append(dataSet[x].append(labelSet[x]))#dude this is broken
        return doneSet

def main():
        with np.load("TINY_MNIST.npz") as data:
            #trainingSet, trainingLabel = data["x"], data["t"]
            trainingLength = 800
            x, t = data["x"][:trainingLength], data["t"][:trainingLength]
            #print repr(len(x))
            x_eval, t_eval = data["x_eval"], data["t_eval"]
        #training and test sets
        print('Train set: ' + repr(len(x)))
        print('Test set: ' + repr(len(x_eval)))
        #generate predictions
        predictions=[]
        correct=0
        k = 401
        for i in range(len(x_eval)):
        #first, try for just first instance test
            neighbours = getNeighbors(x, t, x_eval[i], k)#modify neighbours function?
            #print repr(neighbours)#got three neighbours, 2/3 are "1" so likely a 5
            result = getResponse(neighbours)
            predictions.append(result)#list of strings { '[0.]',...] t_eval is [[0.],...]
            #if predictions[i] == t_eval[i][0] correct += 1
            print('> predicted=' + repr(result) + ', actual=' + repr(t_eval[i][0]))
        accuracy = getAccuracy(x_eval, t_eval, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')
main()