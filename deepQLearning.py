import argparse
import copy
import datetime
import logging
import multiprocessing
import os
import pytz
import random
import re
import time

import numpy as np
import matplotlib.pyplot as plt

import pygame
import snakeGame


def getGameImage():
    # get game screen image
    image = pygame.surfarray.array3d(pygame.display.get_surface())

    # reduce the size of the image into 10x10 pixels and convert it to grayscale
    # keep the green channel to differentiate the snake from the food
    image = image[::snakeGame.GRID_SPACE, ::snakeGame.GRID_SPACE, 1]

    # The is rotated 90 degrees and flipped, so we need to rotate it back
    image = np.rot90(image, -1)
    image = np.flip(image, 1)
    return image


def saveImage(image, filename):
    plt.imsave(filename, image, cmap="gray")


def main():
    game = snakeGame.SnakeGame()
    step = 1
    while True:
        snake = game.snakes[0]
        pygame.event.get()  # Otherwise pygame will freeze
        snake.setDirectionFromMove(random.choice(snakeGame.Move.all))
        game.draw(displayScore=False)

        # save image into grayscale bmp
        saveImage(getGameImage(), "image{}.bmp".format(step))

        step += 1

        if snake.died:
            game.deleteSnakes()
            game.createSnakes()

        pygame.time.wait(1000)
        step += 1


if __name__ == "__main__":
    main()
