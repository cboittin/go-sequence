import os
from random import shuffle
from itertools import islice
import numpy as np
import tensorflow as tf
import simplejson as json
import time

BATCH_SIZE = 128
N_FILTERS = 128
N_LAYERS = 10

MIN_EPOCHS = 8

TRAIN_DATA_PERCENT = 0.95

x = tf.placeholder("float", [None, 4*19*19])
y = tf.placeholder("float", [None, 5*2*20])

keep_prob = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

TF_LOG = os.path.join("model", "tf.log")
NETWORK_CHECKPOINT = os.path.join("model", "model.ckpt")

def weight_variable(shape):
    stddev = np.sqrt(2.0 / (sum(shape)))
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME", data_format="NCHW")

def residualLayer(inputs):
    W_conv1 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
    W_conv2 = weight_variable([3, 3, N_FILTERS, N_FILTERS])
    
    skip_connection = tf.identity(inputs)
    
    conv1 = conv2d(inputs, W_conv1)
    conv1_bn = tf.layers.batch_normalization(conv1, axis=1, fused=True, training=training)
    conv1_out = tf.nn.relu(conv1_bn)

    conv2 = conv2d(conv1_out, W_conv1)
    conv2_bn = tf.layers.batch_normalization(conv2, axis=1, fused=True, training=training)
    
    skip = tf.add(conv2_bn, skip_connection)
    output = tf.nn.relu(skip)
    
    return output

def convolutionalLayer(inputs, input_channels, output_channels):
    W_conv = weight_variable([3, 3, input_channels, output_channels])
    
    conv = conv2d(inputs, W_conv)
    conv_bn = tf.layers.batch_normalization(conv, fused=True, training=training)
    output = tf.nn.relu(conv_bn)
    
    return output
    
def outputLayer(inputs):
    W_fc = weight_variable([10*19*19, 5*2*20])
    b_fc = bias_variable([5*2*20])
    
    conv = convolutionalLayer(inputs, N_FILTERS, 10)
    conv_flat = tf.reshape(conv, [-1, 10*19*19])
    
    fc = tf.nn.relu( tf.matmul(conv_flat, W_fc) + b_fc )
    dropout = tf.nn.dropout(fc, keep_prob)
    
    output = tf.nn.relu(tf.sign(dropout))
    
    return output
    # W_out = weight_variable([5, 2, 20])
    # output = tf.matmul(dropout, weights["out"]) + biases["out"]
    
def neuralNetworkModel(x):
    x = tf.reshape(x, shape=[-1, 4, 19, 19])    # 4 layers of a 19 by 19 board, init black stones, init white stones, final black stones, final white stones

    layer = convolutionalLayer(x, 4, N_FILTERS)   # input layer
    
    for _ in range(N_LAYERS):  # hidden layers
        layer = residualLayer(layer)
        
    output = outputLayer(layer)
    
    return output
    
def train_neural_network(trainData, testData):
    print("Start Training")
    prediction = neuralNetworkModel(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    # optimizer = tf.contrib.opt.NadamOptimizer().minimize(cost)    # Should update the learning rate ? Read the corresponding article http://cs229.stanford.edu/proj2015/054_report.pdf
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Get the epoch from the tf_log, if any
    epoch = 1
    try:
        log = open(TF_LOG, "r").read().split("\n")
        epoch = int(log[-2])+1
        print("Starting at epoch", epoch)
    except:
        print("Starting training")
        
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        if epoch != 1:
            saver.restore(sess, NETWORK_CHECKPOINT)
        
        n_batches_train = trainData.getTotalBatches(BATCH_SIZE)
        n_batches_test = testData.getTotalBatches(BATCH_SIZE)
        
        # modelOutput = tf.split(prediction, BATCH_SIZE)
        # expectedOutput = tf.split(y, BATCH_SIZE)
        correct = tf.reduce_all(tf.equal(prediction, y), axis=1)
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        accuracy_sum = tf.reduce_sum(tf.cast(correct, tf.float32))
        
        def train(epoch):
            print("Training : %d batches" % n_batches_train)
            epoch_loss = 0
            start = time.time()
            for i in range(n_batches_train):
                batch_x, batch_y = trainData.getNextBatch(BATCH_SIZE)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, training: True, keep_prob:0.5})
                epoch_loss += c
                if i % 500 == 499:
                    end = time.time()
                    print("Epoch %d - Batch %d out of %d - %.2fs" % (epoch, i+1, n_batches_train, end - start))
                    start = end
        
            saver.save(sess, NETWORK_CHECKPOINT)
            with open(TF_LOG, "a") as f:
                f.write(str(epoch)+"\n")
            print("Epoch", epoch, "completed with loss:", epoch_loss)
    
        def test():
            print("Testing : %d batches" % n_batches_test)
            cumulative_accuracy = 0.0
            total = 0
            for index in range(n_batches_test):
                batch_x, batch_y = testData.getNextBatch(BATCH_SIZE)
                cumulative_accuracy += accuracy_sum.eval(feed_dict={x: batch_x, y: batch_y, training: False, keep_prob: 1.0})
                total += batch_y.shape[0]
            totalAccuracy = cumulative_accuracy / total
            print("Test accuracy : %f" % totalAccuracy)
            return totalAccuracy
        
        while epoch < MIN_EPOCHS - 1:
            train(epoch)
            epoch += 1
        
        prevAccuracy = test()
        train(epoch)
        epoch += 1
        modelAccuracy = test()
        while modelAccuracy > prevAccuracy:
            train(epoch)
            epoch += 1
            prevAccuracy = modelAccuracy
            modelAccuracy = test()
            
        print("Done training the model.")

class Data:
    
    def __init__(self, bufferSize=10000):
        self.files = {}
        self.weights = {}
        self.bufferSize = bufferSize
        self.totalRecords = 0
        self.inputBuffer = None
        self.outputBuffer = None
        self.currentBufferIndex = 0
        
    def parseData(self, filePaths):
        # Load line counts for the records files
        lcFile = open("linecounts.json")
        lineCounts = json.load(lcFile)
        lcFile.close()
        
        # Create file object handles
        for filePath in filePaths:
            fileObject = open(filePath, "r", buffering=self.bufferSize)
            lines = lineCounts[filePath]
            self.totalRecords += lines
            self.files[filePath] = (fileObject, lines, 0)   # 3rd value is the current line index in the file
            self.weights[filePath] = lines
            print("Database %s : %d records" % (filePath.rsplit("/", 1)[1], lines))
            
        # Calculate weights for each file
        sumLines = 0
        for w in self.weights.values():
            sumLines += w
        for k, v in self.weights.items():
            w = float(v) / sumLines
            self.weights[k] = w
            print("Database %s : weight %f" % (k.rsplit("/", 1)[1], w))
        
        self.loadData()
        
    def loadData(self):
        inBuffer = [None]*self.bufferSize
        outBuffer = [None]*self.bufferSize
        total = 0
        for filePath, db in self.files.items():
            dbObject, totalLines, lineCount = db
            
            picks = int(self.weights[filePath] * self.bufferSize) # Number of records to pick
            if picks + lineCount > totalLines:  # EOF check
                picks = totalLines - lineCount
                
            for i in range(total, total + picks):
                record = self.parseRecord(dbObject.readline())
                inBuffer[i] = record[0]
                outBuffer[i] = record[1]
            
            self.files[filePath] = (dbObject, totalLines, lineCount + picks)
            total += picks
        
        if total < self.bufferSize:
            self.inputBuffer = inBuffer[:total]
            self.outputBuffer = outBuffer[:total]
        else:
            self.inputBuffer = inBuffer
            self.outputBuffer = outBuffer
        shuffle(self.inputBuffer)
        shuffle(self.outputBuffer)
        self.currentBufferIndex = 0
        # print("Loading %d new records" % total)
        
    def parseRecord(self, record):
        i, o = record.split(":")
        input = np.zeros(4*19*19)
        for index in range(len(input)):
            if i[index] == "1":
                input[index] = 1
        output = np.zeros(5*2*20)
        for index in range(len(output)):
            if o[index] == "1":
                output[index] = 1
        return (input, output)
        
    def resetFiles(self):
        print("Resetting database indices.")
        for filePath in self.files.keys():
            self.resetFile(filePath)
        
    def resetFile(self, filePath):
        db = self.files[filePath]
        dbObject, totalLines, _ = db
        dbObject.close()
        dbObject = open(filePath, "r", buffering=self.bufferSize)
        self.files[filePath] = (dbObject, totalLines, 0)
    
    def getTotalBatches(self, batch_size):
        return int(self.totalRecords / batch_size)
    
    def getNextBatch(self, batch_size):
        start = self.currentBufferIndex
        end = start + batch_size
        batch_x = None
        batch_y = None
        
        if end <= len(self.inputBuffer):
            batch_x = self.inputBuffer[start:end]
            batch_y = self.outputBuffer[start:end]
        else:  # We need to load more data
            batch_x = self.inputBuffer[start:]
            batch_y = self.outputBuffer[start:]
            
            self.loadData()
            end = batch_size - len(batch_x)
            
            if end > len(self.inputBuffer): # End of training data
                batch_x += self.inputBuffer
                batch_y += self.outputBuffer
                self.resetFiles()
                self.loadData()
                return ( np.array(batch_x), np.array(batch_y) )
            else:
                batch_x = self.inputBuffer[:end]
                batch_y = self.outputBuffer[:end]
        
        self.currentBufferIndex = end
        return ( np.array(batch_x), np.array(batch_y) )
        
        
if __name__ == "__main__":
    trainData = Data(10000)
    trainFolder = "/".join( ("D:", "Projects", "NeuralNets", "Go", "train") )
    databases = []
    for database in os.listdir(trainFolder):
        fullPath = "/".join( (trainFolder, database) )
        databases.append(fullPath)
    trainData.parseData(databases)
    
    testData = Data(2000)
    testFolder = "/".join( ("D:", "Projects", "NeuralNets", "Go", "test") )
    databases = []
    for database in os.listdir(testFolder):
        fullPath = "/".join( (testFolder, database) )
        databases.append(fullPath)
    testData.parseData(databases)
    
    train_neural_network(trainData, testData)
