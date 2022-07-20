import rasa_nlu
import rasa_core
import spacy

import warnings
warnings.filterwarnings("ignore")

import logging,io, json, warnings
logging.basicConfig(level="INFO")
warnings.filterwarnings('ignore')


# Import modules for training
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config

# loading the nlu training samples
training_data = load_data("nlu.md")
trainer = Trainer(config.load("config.yml"))

# training the nlu
interpreter = trainer.train(training_data)
model_directory = trainer.persist("./models/nlu", fixed_model_name="current")

# Import the policies and agent
from rasa_core.policies import FallbackPolicy, MemoizationPolicy,KerasPolicy
from rasa_core.agent import Agent

# Initialize the model with `domain.yml`
agent = Agent('domain.yml', policies=[MemoizationPolicy(), KerasPolicy()])

# loading our  training dialogues from `stories.md`
training_data = agent.load_data('stories.md')


# Training the model
agent.train(
    training_data,
    validation_split=0.1,
    epochs=150
)

agent.persist('models/dialogue')

#Starting the Bot

from rasa_core.agent import Agent
agent = Agent.load('models/dialogue', interpreter=model_directory)

print("Hello, Zachary! How may I help you?")
while True:
    a = input()
    if a == 'stop':
        break
    responses = agent.handle_message(a)
    for response in responses:
        print(response["text"])
