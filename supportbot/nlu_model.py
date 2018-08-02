from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter
import pandas as pd
import numpy as np

def train_nlu(data, configs, model_dir):
	training_data = load_data(data)
	trainer = Trainer(config.load(configs))
	trainer.train(training_data)
	model_directory = trainer.persist(model_dir, fixed_model_name = 'supportnlu')
	
def run_nlu():
	interpreter = Interpreter.load('./models/nlu/default/supportnlu')
	print(interpreter.parse("MySpace Password Expiry")['intent']['name'])

	dfOutput = pd.read_csv('SampleOutput.csv')
	dfAns = pd.read_csv('Answers.csv')
	dictIntentMap = dict(zip(dfAns.intent,dfAns.Answers))
	for index1,row1 in dfOutput.iterrows():
		reply = interpreter.parse(row1['Question'])['intent']['name']
		dfOutput.ix[index1,"Correct"]=(dictIntentMap.get(reply, "No Answer Found")==row1['Answers'])
		dfOutput.ix[index1,"Reply"]=dictIntentMap.get(reply, "No Answer Found")
	print("Test Set Accuracy:{0:.2%}".format(np.sum(dfOutput['Correct'])/dfOutput.shape[0]))
	dfOutput.to_csv('OutputReply.csv')

def run_nlu_train():
	interpreter = Interpreter.load('./models/nlu/default/supportnlu')
	dfInput = pd.read_csv('SampleInput.csv')
	dfAns = pd.read_csv('Answers.csv')
	dictIntentMap = dict(zip(dfAns.intent,dfAns.Answers))	
	for index1,row1 in dfInput.iterrows():
		reply = interpreter.parse(row1['Question'])['intent']['name']
		dfInput.ix[index1,"Reply"]= dictIntentMap.get(reply, "No Answer Found")
		dfInput.ix[index1,"Correct"]=(dictIntentMap.get(reply, "No Answer Found")==row1['Answers'])
	print("Train Set Accuracy:{0:.2%}".format(np.sum(dfInput['Correct'])/dfInput.shape[0]))
	dfInput.to_csv('InputReply.csv')

if __name__ == '__main__':
	#train_nlu('./data/nlu.md', 'config3.yaml', './models/nlu')
	run_nlu()
	run_nlu_train()