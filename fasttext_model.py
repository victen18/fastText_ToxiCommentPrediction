import ktrain
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#use only CPU for precition

predictor = ktrain.load_predictor('toxic_fasttext')

def get_prediction(x):

	pred = predictor.predict([x])
	return pred[0]

