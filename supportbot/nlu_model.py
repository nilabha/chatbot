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
	#print(interpreter.parse("MySpace Password Expiry")['intent']['name'])

	dfOutput = pd.read_csv('SampleOutput.csv')
	dfAns = pd.read_csv('Answers.csv')
	dictIntentMap = dict(zip(dfAns.intent,dfAns.Answers))	
	#print(interpreter.parse("Login Page is not getting displayed"))
	#print(interpreter.parse("Forgot the password, can you please reset the password for me."))	
	for index1,row1 in dfOutput.iterrows():
		reply = interpreter.parse(row1['Question'])['intent']['name']
		#print(interpreter.parse(row1['Question']))
		dfOutput.ix[index1,"Correct"]=(dictIntentMap.get(reply, "")==row1['Answers'])
		dfOutput.ix[index1,"Reply"]=dictIntentMap.get(reply, "")
	print("Test Set Accuracy:{0:.2%}".format(np.sum(dfOutput['Correct'])/dfOutput.shape[0]))
	dfOutput.to_csv('OutputReply.csv')

def run_nlu_train():
	interpreter = Interpreter.load('./models/nlu/default/supportnlu')
	#print(interpreter.parse("MySpace Password Expiry")['intent']['name'])

	dfInput = pd.read_csv('SampleInput.csv')
	dfAns = pd.read_csv('Answers.csv')
	dictIntentMap = dict(zip(dfAns.intent,dfAns.Answers))	
	#print(interpreter.parse("Login Page is not getting displayed"))
	#print(interpreter.parse("Forgot the password, can you please reset the password for me."))	
	for index1,row1 in dfInput.iterrows():        
		#print(interpreter.parse(row1['Question']))

		reply = interpreter.parse(row1['Question'])['intent']['name']
		#print("reply:",dictIntentMap[reply])
		#print("answer:",row1['Answers'])
		#print(dictIntentMap[reply]==row1['Answers'])
		dfInput.ix[index1,"Reply"]= dictIntentMap.get(reply, "")
		dfInput.ix[index1,"Correct"]=(dictIntentMap.get(reply, "")==row1['Answers'])
	print("Train Set Accuracy:{0:.2%}".format(np.sum(dfInput['Correct'])/dfInput.shape[0]))
	dfInput.to_csv('InputReply.csv')

if __name__ == '__main__':
	#train_nlu('./data/nlu.md', 'config3.yaml', './models/nlu')
	run_nlu()
	run_nlu_train()
