import datetime
import pytz
import os
import traceback
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import re
import time
import argparse
import random
import copy
import multiprocessing
import logging

import numpy as np
import matplotlib.pyplot as plt
import pygame
import tensorflow as tf
from keras import layers
from keras import models

import snakeGame

LOGGER = logging.getLogger()
LOG_FORMATTER = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(LOG_FORMATTER)
LOGGER.addHandler(consoleHandler)
LOGGER.setLevel(logging.INFO)

def tracebackHandler(type, value, tb):
    for line in traceback.TracebackException(type, value, tb).format(chain=True):
        LOGGER.exception(line)
    LOGGER.exception(value)

    sys.__excepthook__(type, value, tb)  # In order to have the traceback in stdout

sys.excepthook = tracebackHandler

LOGGER.info('Starting training')
RESULTS_DIR = 'results'

AGENT_POPULATION_COUNT = 500
MAX_GENERATIONS = 500
MAX_STEPS_FOR_NEGATIVE_SCORE = 50
MAX_STEPS = 1000

INPUT_SIZE = 16
DENSE_1_SIZE = 12
OUTPUT_SIZE = 3
assert OUTPUT_SIZE == len(snakeGame.Move.all)
assert INPUT_SIZE == snakeGame.STATE_SIZE

MUTATION_RATE = 0.05
MUTATION_WEIGHTS_SIGMA = 0.3
MUTATION_BIASES_SIGMA = 0.05

PROCESS_NB = int(os.cpu_count() * 1.5)
BEST_AGENT_SELECTION_NUMBER = 6
AGENTS_NUMBER_RETURNED_BY_PROCESS = 20

REWARD_APPLE_EATEN = 50
REWARD_DEATH = -100
REWARD_STEP_TOWARDS_APPLE = 0.5
REWARD_STEP_AWAY_FROM_APPLE = -1

MAX_STEPS_WITHOUT_EATING = 50  # if the snake doesn't eat for this amount of steps, it receives a negative reward
REWARD_STEPS_WITHOUT_EATING = -20
LOOP_REWARD = False
LOOP_DEATH = True

SETTINGS_TO_SAVE = {
    'AGENT_POPULATION_COUNT': AGENT_POPULATION_COUNT,
    'MAX_GENERATIONS': MAX_GENERATIONS,
    'MAX_STEPS_FOR_NEGATIVE_SCORE': MAX_STEPS_FOR_NEGATIVE_SCORE,
    'MAX_STEPS': MAX_STEPS,
    'INPUT_SIZE': INPUT_SIZE,
    'DENSE_1_SIZE': DENSE_1_SIZE,
    'OUTPUT_SIZE': OUTPUT_SIZE,
    'MUTATION_RATE': MUTATION_RATE,
    'MUTATION_WEIGHTS_SIGMA': MUTATION_WEIGHTS_SIGMA,
    'MUTATION_BIASES_SIGMA': MUTATION_BIASES_SIGMA,
    'PROCESS_NB': PROCESS_NB,
    'BEST_AGENT_SELECTION_NUMBER': BEST_AGENT_SELECTION_NUMBER,
    'AGENTS_NUMBER_RETURNED_BY_PROCESS': AGENTS_NUMBER_RETURNED_BY_PROCESS,
    'REWARD_APPLE_EATEN': REWARD_APPLE_EATEN,
    'REWARD_DEATH': REWARD_DEATH,
    'REWARD_STEP_TOWARDS_APPLE': REWARD_STEP_TOWARDS_APPLE,
    'REWARD_STEP_AWAY_FROM_APPLE': REWARD_STEP_AWAY_FROM_APPLE,
    'MAX_STEPS_WITHOUT_EATING': MAX_STEPS_WITHOUT_EATING,
    'REWARD_STEPS_WITHOUT_EATING': REWARD_STEPS_WITHOUT_EATING,
    'LOOP_REWARD': LOOP_REWARD,
    'LOOP_DEATH': LOOP_DEATH,
}

logging.basicConfig(level=logging.ERROR)


class Agent:
    def __init__(self, game=None):
        if game:
            self.snake = game.snakes[0]
            self.apple = game.apples[0]

        self.fitness = 0
        self.distanceTraveled = 0

        self.model = models.Sequential()
        self.model.add(layers.Input(shape=(1, INPUT_SIZE)))

        self.model.add(
            layers.Dense(
                units=DENSE_1_SIZE,
                activation="relu",
                bias_initializer="random_uniform",
            )
        )

        self.model.add(
            layers.Dense(
                units=OUTPUT_SIZE,
                activation="softmax",
                bias_initializer="random_uniform",
            )
        )

        # in order to set the weights
        self.model.build(input_shape=(None, INPUT_SIZE))
        self.predictedMove = None
        self.step = 0
        self.stepAte = 0

    def incrementFitness(self):
        if self.snake.died:
            self.fitness += REWARD_DEATH
            return

        if self.snake.justAte:
            self.fitness += REWARD_APPLE_EATEN
            self.stepAte = self.step

        if self.snake.isGoingTowardsTheApple(self.apple):
            self.fitness += REWARD_STEP_TOWARDS_APPLE
        else:
            self.fitness += REWARD_STEP_AWAY_FROM_APPLE

        if LOOP_REWARD:
            if self.step % 10 == 0:
                if self.step - self.stepAte > MAX_STEPS_WITHOUT_EATING:
                    self.fitness += REWARD_STEPS_WITHOUT_EATING

        if LOOP_DEATH:
            if self.step > 200:
                if self.step - self.stepAte > MAX_STEPS_WITHOUT_EATING:
                    self.snake.died = True
                    return

        self.step += 1

    def save(self, model="model.h5"):
        self.model.save(model)

    def saveModelIfBest(self, cacheDir):

        if self.fitness < 0:
            return

        regex = re.compile(r'bestAgent_(\d+).h5')
        savedModels = {}

        for model in os.listdir(cacheDir):
            match = regex.match(model)
            if not match:
                continue
            savedModels[int(match.group(1))] = model

        modelPath = os.path.join(cacheDir, f'bestAgent_{int(self.fitness)}.h5')
        if len(savedModels) <= 10:
            self.model.save(modelPath)
        else:
            if self.fitness > min(savedModels.keys()):
                os.remove(os.path.join(cacheDir, savedModels[min(savedModels.keys())]))
                self.model.save(modelPath)

    def load(self, model):
        self.model.load_weights(model)

    def getWeights(self):
        return self.model.get_weights()

    def setWeights(self, weights):
        self.model.set_weights(weights)

    def predict(self, state):
        inputs = np.array([[state]])
        prediction = self.model.predict(inputs, verbose=0)
        return snakeGame.Move.all[prediction.argmax()]

    @staticmethod
    def crossOver(weights1, weights2):
        newWeights = copy.deepcopy(weights1)
        for i in range(len(newWeights)):

            # biases
            if len(newWeights[i].shape) != 2:
                for j in range(len(newWeights[i])):
                    randValue = random.random()
                    if randValue < 0.5:
                        newWeights[i][j] = weights2[i][j]
                continue

            # weights
            for j in range(len(newWeights[i])):
                for k in range(len(newWeights[i][j])):
                    randValue = random.random()
                    if randValue < 0.5:
                        newWeights[i][j][k] = weights2[i][j][k]

        return newWeights

    @staticmethod
    def mutateWeights(weights, rate=MUTATION_RATE):
        newWeights = copy.deepcopy(weights)
        for layerWeights in newWeights:

            # biases
            if len(layerWeights.shape) != 2:
                for k in range(len(layerWeights)):
                    randValue = random.random()
                    if randValue < rate:
                        layerWeights[k] = random.gauss(mu=0, sigma=MUTATION_BIASES_SIGMA)
                continue

            # weights
            for i in range(len(layerWeights)):
                for j in range(len(layerWeights[i])):
                    randValue = random.random()
                    if randValue < rate:
                        layerWeights[i][j] = random.gauss(mu=0, sigma=MUTATION_WEIGHTS_SIGMA)

        return newWeights


def getLastCachedModels():
    regex = re.compile(r'bestAgent_(\d+).h5')
    savedModels = []

    for dateDir in sorted(os.listdir(RESULTS_DIR), reverse=True):
        dateDirPath = os.path.join(RESULTS_DIR, dateDir)
        foundModels = False
        for model in sorted(os.listdir(dateDirPath), reverse=True):
            match = regex.match(model)
            if match:
                savedModels.append(os.path.join(dateDirPath, model))
                foundModels = True

        if foundModels:
            return savedModels


def testModels(cachedModels=None, useCachedModels=False):
    if useCachedModels:
        cachedModels = getLastCachedModels()

    maxStep = 500

    for model in cachedModels:
        LOGGER.info(f'Testing model: {model}')
        game = snakeGame.SnakeGame(appleCount=1, snakeCount=1)
        agent = Agent(game)
        agent.load(model)

        keepPlaying = True

        step = 1
        while keepPlaying:
            agent.snake = game.snakes[0]  # because of the reset

            # skip to next agent with space bar
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        keepPlaying = False

            snake = agent.snake
            state = snake.getState(game.apples)

            prediction = agent.predict(state)
            snake.setDirectionFromMove(prediction)

            game.draw(snakeReset=True)
            if step >= maxStep:
                keepPlaying = False
                LOGGER.info('Max step reached')
            if snake.died:
                keepPlaying = False


def plotGraph(genCounter, fitnessValues):
    plt.scatter(
        x=[[i] * len(fitnessValues[0]) for i in range(1, genCounter + 1)],
        y=fitnessValues,
        c="blue",
    )

    average = [sum(i) / len(i) for i in fitnessValues]
    gen = [i for i in range(1, genCounter + 1)]
    plt.plot(gen, average, c="red")

    plt.legend(["Fitness values", "Average fitness"])
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.savefig(f"{CACHE_DIR}/learningCurveGraph.jpg")


def gameLoop(game, agent):
    step = 1
    while True:
        pygame.event.get()  # Otherwise pygame will freeze

        snake = agent.snake
        state = snake.getState(game.apples)

        prediction = agent.predict(state)
        snake.setDirectionFromMove(prediction)
        agent.incrementFitness()

        game.draw(hudInfo=f"Fitness: {agent.fitness}")

        if snake.died or (step > MAX_STEPS_FOR_NEGATIVE_SCORE and agent.fitness < 0) or step > MAX_STEPS:
            game.deleteSnakes()
            game.createSnakes()
            break

        step += 1


def trainGeneration(procId, population, bestWeights, bestFitness, returnData, cacheDir, initCachedModel=None):

    # game windows in line on top of the screen
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{250 * procId},0"

    game = snakeGame.SnakeGame()

    agents = []
    for i in range(population):
        agent = Agent(game)

        if initCachedModel and not bestWeights:
            agent.load(initCachedModel)
            newWeights = agent.mutateWeights(agent.getWeights())
            agent.setWeights(newWeights)

        if bestWeights:
            # select randomly the weights based on their fitness, crossover and mutate
            selectedWeights = random.choices(bestWeights, weights=bestFitness, k=2)
            crossedOverWeights = agent.crossOver(selectedWeights[0], selectedWeights[1])
            newWeights = agent.mutateWeights(crossedOverWeights)
            agent.setWeights(newWeights)

        gameLoop(game, agent)
        agents.append(agent)

    # return an arbitrary number of the best agents that will be selected by the main process
    sortedAgents = sorted(agents, key=lambda x: x.fitness, reverse=True)[:AGENTS_NUMBER_RETURNED_BY_PROCESS]
    returnData[procId] = [agent.getWeights() for agent in sortedAgents], [agent.fitness for agent in sortedAgents]

    sortedAgents[0].saveModelIfBest(cacheDir)

    tf.keras.backend.clear_session()  # to avoid having memory leak


def main(model=None, cacheDir=None):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    fileHandler = logging.FileHandler(f"{CACHE_DIR}/resultLog.log")
    fileHandler.setFormatter(LOG_FORMATTER)
    LOGGER.addHandler(fileHandler)

    LOGGER.info('Starting training')

    with open(f"{CACHE_DIR}/settings.txt", "w") as f:
        for key, value in SETTINGS_TO_SAVE.items():
            f.write(f"{key}: {value}\n")

    bestWeightsPerGen = []
    bestFitnessPerGen = []
    fitnessToGraph = []

    for genCounter in range(1, MAX_GENERATIONS + 1):
        t1 = time.time()
        LOGGER.info(f'Generation {genCounter}')

        manager = multiprocessing.Manager()
        returnData = manager.dict()

        procs = [
            multiprocessing.Process(
                target=trainGeneration,
                args=(
                    procId,
                    AGENT_POPULATION_COUNT // PROCESS_NB,
                    bestWeightsPerGen,
                    bestFitnessPerGen,
                    returnData,
                    cacheDir,
                    model,
                ),
            )
            for procId in range(PROCESS_NB)
        ]

        for i in procs:
            i.start()

        for i in procs:
            i.join()

        weightsPerGen = []
        fitnessPerGen = []
        for i in range(PROCESS_NB):
            weights, fitness = returnData[i]
            weightsPerGen.extend(weights)
            fitnessPerGen.extend(fitness)

        bestFitnessPerGen = sorted(fitnessPerGen, reverse=True)[:BEST_AGENT_SELECTION_NUMBER]

        # selection of the best agents
        bestWeightsPerGen = []
        for weights, fitness in zip(weightsPerGen, fitnessPerGen):
            if fitness in bestFitnessPerGen and len(bestWeightsPerGen) < len(bestFitnessPerGen):
                bestWeightsPerGen.append(weights)

        LOGGER.info(f'Best fitness: {bestFitnessPerGen}')

        fitnessToGraph.append(fitnessPerGen)
        plotGraph(genCounter, fitnessToGraph)
        LOGGER.info(f'Generation {genCounter} done in {time.time() - t1} seconds')


if __name__ == "__main__":
    # create cache dir based on datetime
    CACHE_DIR = f"results/{datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d_%H-%M-%S')}"

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--model", type=str, help="model to load for training or testing")
    argParser.add_argument(
        "--initWithBestCachedModel", action='store_true', help="use the best cached model for training"
    )
    argParser.add_argument("--test", action="store_true", help="test the given model")
    argParser.add_argument("--testCached", action="store_true", help="test the cached models in `CACHE_DIR` folder")

    args = argParser.parse_args()
    if args.test:
        if not args.model:
            raise Exception("No model given for testing")
        testModels([args.model], useCachedModels=False)

    elif args.testCached:
        testModels(useCachedModels=True)

    elif args.initWithBestCachedModel:
        cachedModel = sorted(getLastCachedModels())[-1]
        LOGGER.info(f'Use best cached model for init: {cachedModel}')
        main(model=cachedModel, cacheDir=CACHE_DIR)

    else:
        main(model=args.model, cacheDir=CACHE_DIR)
