



from imageai.Prediction import ImagePrediction
import os

'''

https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5


'''

execution_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsResNet()

model_file = os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5")
prediction.setModelPath(model_file)
prediction.loadModel()
predictions, percentage_probabilities = prediction.predictImage("image.jpg", result_count=5)

for index in range(len(predictions)):
    print('%s: %s' % (predictions[index], percentage_probabilities[index]))
