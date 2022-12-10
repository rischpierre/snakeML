import os
import random
import pygame
import numpy as np

DISPLAY_GRID_SIZE = (250, 250)
DISPLAY_GRID_SIZE_DEBUG = (500, 250)
SNAKE_GRID_SIZE_ABSOLUTE = (250, 250)
GRID_SPACE = 25
assert SNAKE_GRID_SIZE_ABSOLUTE[0] % GRID_SPACE == 0
assert SNAKE_GRID_SIZE_ABSOLUTE[1] % GRID_SPACE == 0

SPEED = 2
SNAKE_GRID_SIZE_RELATIVE = (
    int(SNAKE_GRID_SIZE_ABSOLUTE[0] / GRID_SPACE),
    int(SNAKE_GRID_SIZE_ABSOLUTE[1] / GRID_SPACE),
)
GRID_COLOR = (25, 25, 25)
FOOD_COLOR = (200, 100, 25)
BG_COLOR = (20, 20, 20)

# size of the danger grid and relatives positions of the apple in +x, -x, +y, -y
DANGER_GRID_SIZE = 3
STATE_SIZE = (DANGER_GRID_SIZE * DANGER_GRID_SIZE) - 1 + 8


class Dir:
    left = 0
    right = 1
    up = 2
    down = 3


class Move:
    left = 0
    right = 1
    forward = 2
    all = [left, right, forward]


class Apple:
    def __init__(self, game):
        self.screen = game.screen
        self.game = game
        self.position = (0, 0)
        self.getRandomPosition()

    def getRandomPosition(self):

        self.position = (
            random.choice([i for i in self.game.wholeGrid if i not in self.game.snakes[0].bodyQueue])
        )

    def draw(self):
        pygame.draw.rect(
            surface=self.screen,
            color=FOOD_COLOR,
            rect=(
                self.position[0] * GRID_SPACE,
                self.position[1] * GRID_SPACE,
                GRID_SPACE,
                GRID_SPACE,
            ),
        )

    def respawn(self):
        self.getRandomPosition()
        self.draw()


class Snake:
    def __init__(self, screen, randomInit=False):
        self.screen = screen
        self.initLength = 4
        self.length = self.initLength
        self.defaultColor = (200, 200, 200, 200)
        self.color = self.defaultColor
        self.height = GRID_SPACE
        self.width = GRID_SPACE
        self.direction = Dir.right
        self.died = False
        self.justAte = False
        self.headPosition = self.setStartHeadPosition(randomInit)
        self.stateSize = None
        self.bodyQueue = self.getStartBodyQueue()

    @staticmethod
    def setStartHeadPosition(randomInit=False) -> "list[int]":
        if randomInit:
            return [
                random.randint(1, SNAKE_GRID_SIZE_RELATIVE[0] - 1),
                random.randint(1, SNAKE_GRID_SIZE_RELATIVE[1] - 1),
            ]

        return [
            int(SNAKE_GRID_SIZE_RELATIVE[0] / 2),
            int(SNAKE_GRID_SIZE_RELATIVE[1] / 2),
        ]

    def getStartBodyQueue(self):
        bodyQueue = [self.headPosition.copy()]
        for i in range(0, self.length):
            bodyQueue.append([bodyQueue[-1][0] - 1, bodyQueue[-1][1]])
        return bodyQueue

    def reset(self):
        self.length = self.initLength
        self.headPosition = self.setStartHeadPosition()
        self.bodyQueue = self.getStartBodyQueue()

    def draw(self) -> bool:

        for bodyElement in self.bodyQueue:
            pygame.draw.rect(
                surface=self.screen,
                color=self.color,
                rect=(
                    bodyElement[0] * GRID_SPACE,
                    bodyElement[1] * GRID_SPACE,
                    self.height,
                    self.width,
                ),
            )
        return False

    def isDead(self) -> bool:
        for i in range(2):
            # check if snake hit the corners
            if self.headPosition[i] >= SNAKE_GRID_SIZE_RELATIVE[i]:
                self.died = True
                break

            elif self.headPosition[i] < 0:
                self.died = True
                break

        if self.headPosition in self.bodyQueue[1:]:
            self.died = True

        return self.died

    def eat(self, apples):
        self.justAte = False

        for apple in apples:
            if self.headPosition[0] == apple.position[0] and self.headPosition[1] == apple.position[1]:
                apple.respawn()
                self.length += 1
                self.bodyQueue.append(self.bodyQueue[-1])
                self.justAte = True
                return True

    def move(self):

        if self.direction == Dir.right:
            self.headPosition[0] += 1

        elif self.direction == Dir.down:
            self.headPosition[1] += 1

        elif self.direction == Dir.left:
            self.headPosition[0] -= 1

        elif self.direction == Dir.up:
            self.headPosition[1] -= 1

        self.bodyQueue.insert(0, self.headPosition.copy())
        self.bodyQueue.pop(-1)

    def getDangerGrid(self):
        # Defines a size x size grid on the snake to see if there is danger around
        headPosition = self.headPosition
        grid = np.zeros((DANGER_GRID_SIZE, DANGER_GRID_SIZE))
        offset = DANGER_GRID_SIZE // 2

        for x, gridX in enumerate(range(headPosition[0] - offset, headPosition[0] + offset + 1)):
            for y, gridY in enumerate(range(headPosition[1] - offset, headPosition[1] + offset + 1)):
                danger = 0
                if not 0 <= gridX < SNAKE_GRID_SIZE_RELATIVE[0] or not 0 <= gridY < SNAKE_GRID_SIZE_RELATIVE[1]:
                    danger = 1
                for bodyItem in self.bodyQueue:
                    if bodyItem[0] == gridX and bodyItem[1] == gridY:
                        danger = 1
                        break
                grid[x][y] = danger

        # orient the direction of the matrix to the direction of the snake
        if self.direction == Dir.up:
            return grid
        elif self.direction == Dir.left:
            return np.rot90(grid, 1)
        elif self.direction == Dir.down:
            return np.rot90(grid, 2)
        else:
            return np.rot90(grid, 3)

    def isGoingTowardsTheApple(self, apple):
        if self.direction == Dir.right:
            return apple.position[0] > self.headPosition[0]

        elif self.direction == Dir.down:
            return apple.position[1] > self.headPosition[1]

        elif self.direction == Dir.left:
            return apple.position[0] < self.headPosition[0]

        elif self.direction == Dir.up:
            return apple.position[1] < self.headPosition[1]

    def getState(self, apples):
        # size of the danger grid
        size = DANGER_GRID_SIZE * DANGER_GRID_SIZE
        dangerGrid = self.getDangerGrid().reshape(size).tolist()
        dangerGrid.pop(size // 2)  # remove the head of the snake because it is not a danger

        nearestApple = self.getNearestApple(apples)

        # relatives positions of the apple in +x, -x, +y, -y
        relativePosX = (self.headPosition[0] - nearestApple.position[0]) / SNAKE_GRID_SIZE_RELATIVE[0]
        relativePosY = (self.headPosition[1] - nearestApple.position[1]) / SNAKE_GRID_SIZE_RELATIVE[1]

        state = [
            1 if self.direction == Dir.right else 0,  # +x head
            1 if self.direction == Dir.left else 0,  # -x head
            1 if self.direction == Dir.down else 0,  # +y head
            1 if self.direction == Dir.up else 0,  # -y head
            relativePosX if relativePosX > 0 else 0,  # +x apple
            -relativePosX if relativePosX < 0 else 0,  # -x apple
            relativePosY if relativePosY > 0 else 0,  # +y apple
            -relativePosY if relativePosY < 0 else 0,  # -y apple
        ]

        return state + dangerGrid

    def getNearestApple(self, apples) -> Apple:
        nearestApple = None
        nearestDistance = 1000
        for apple in apples:
            distance = abs(self.headPosition[0] - apple.position[0]) + abs(self.headPosition[1] - apple.position[1])
            if distance < nearestDistance:
                nearestApple = apple
                nearestDistance = distance
        return nearestApple or apples[0]

    def setDirectionFromMove(self, move: Move):
        if self.direction == Dir.down:
            if move == Move.right:
                self.direction = Dir.left
            elif move == Move.left:
                self.direction = Dir.right

        elif self.direction == Dir.up:
            self.direction = move

        elif self.direction == Dir.right:
            if move is Move.right:
                self.direction = Dir.down
            elif move == Move.left:
                self.direction = Dir.up

        elif self.direction == Dir.left:
            if move == Move.left:
                self.direction = Dir.down
            elif move == Move.right:
                self.direction = Dir.up


class SnakeGame:
    def __init__(self, snakeCount=1, appleCount=1, randomInit=False, displayDebugScreen=False):
        pygame.init()

        self.wholeGrid = []
        for x in range(0, SNAKE_GRID_SIZE_RELATIVE[0]):
            for y in range(0, SNAKE_GRID_SIZE_RELATIVE[1]):
                self.wholeGrid.append((x, y))

        self.score = 0
        self.screenSize = DISPLAY_GRID_SIZE_DEBUG if displayDebugScreen else DISPLAY_GRID_SIZE

        try:
            self.screen = pygame.display.set_mode(self.screenSize)
        except pygame.error:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            self.screen = pygame.display.set_mode(self.screenSize)

        self.snakeCount = snakeCount
        self.randomInit = randomInit
        self.debugScreen = displayDebugScreen

        self.snakes = None
        self.createSnakes()
        self.apples = [Apple(self) for _ in range(appleCount)]
        self.gameOver = False
        self.paused = False

    def displayDebugScreen(self, screen):
        if not self.snakes:
            return

        # Debug screen
        debugRect = (
            SNAKE_GRID_SIZE_ABSOLUTE[0],
            1,
            self.screenSize[0],
            self.screenSize[1],
        )
        pygame.draw.rect(
            surface=screen,
            color=(240, 240, 240),
            rect=(debugRect[0] + 1, debugRect[1], debugRect[2], debugRect[3] - 1),
            width=2,
        )
        # Danger grid
        dangerRect = (
            debugRect[0] + 1,
            debugRect[1] + 1,
            5 * GRID_SPACE,
            5 * GRID_SPACE,
        )
        pygame.draw.rect(
            surface=screen,
            color=(240, 240, 240),
            rect=(dangerRect[0] + 1, dangerRect[1], dangerRect[2], dangerRect[3] - 1),
            width=2,
        )
        dangerGrid = self.snakes[0].getDangerGrid()

        for x, row in enumerate(dangerGrid):
            for y, _ in enumerate(row):
                if dangerGrid[x][y] == 1:
                    color = (255, 255, 255) if (x == 1 and y == 1) else (200, 200, 200)
                    pygame.draw.rect(
                        surface=screen,
                        color=color,
                        rect=(
                            dangerRect[0] + 1 + x * GRID_SPACE,
                            1 + y * GRID_SPACE,
                            GRID_SPACE,
                            GRID_SPACE,
                        ),
                    )
        # draw the state
        stateRect = (
            dangerRect[0],
            dangerRect[3],
            dangerRect[2],
            dangerRect[3] + 6 * GRID_SPACE,
        )
        pygame.draw.rect(
            surface=screen,
            color=(0, 240, 240),
            rect=(stateRect[0], stateRect[1], stateRect[2], stateRect[3]),
            width=2,
        )

        state = self.snakes[0].getState(self.apples)
        info = [
            f"Head +x {state[0]}",
            f"Head -x {state[1]}",
            f"Head +y {state[2]}",
            f"Head -y {state[3]}",
            f"Appl +x {state[4]}",
            f"Appl -x {state[5]}",
            f"Appl +y {state[6]}",
            f"Appl -y {state[7]}",
        ]
        for i, text in enumerate(info):
            self.displayInfo(text, position=(stateRect[0] + 40, 20 + stateRect[1] + 10 * i), size=10)

    @staticmethod
    def displayGrid(screen):

        for x in range(0, SNAKE_GRID_SIZE_ABSOLUTE[0], GRID_SPACE):
            pygame.draw.line(
                surface=screen,
                color=GRID_COLOR,
                start_pos=(x, 0),
                end_pos=(x, SNAKE_GRID_SIZE_ABSOLUTE[0]),
            )

        for y in range(0, SNAKE_GRID_SIZE_ABSOLUTE[1], GRID_SPACE):
            pygame.draw.line(
                surface=screen,
                color=GRID_COLOR,
                start_pos=(0, y),
                end_pos=(SNAKE_GRID_SIZE_ABSOLUTE[1], y),
            )

        pygame.draw.rect(
            surface=screen,
            color=(240, 240, 240),
            rect=(
                1,
                1,
                SNAKE_GRID_SIZE_ABSOLUTE[0] - 1,
                SNAKE_GRID_SIZE_ABSOLUTE[1] - 1,
            ),
            width=2,
        )

    def createSnakes(self):
        self.snakes = [Snake(self.screen, self.randomInit) for _ in range(self.snakeCount)]

    def deleteSnakes(self):
        self.snakes = []

    def moveOnEvent(self):
        snake = self.snakes[0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                if event.key == pygame.K_RIGHT and snake.direction in (
                    Dir.up,
                    Dir.down,
                ):
                    snake.direction = Dir.right

                elif event.key == pygame.K_LEFT and snake.direction in (
                    Dir.up,
                    Dir.down,
                ):
                    snake.direction = Dir.left

                elif event.key == pygame.K_UP and snake.direction in (
                    Dir.left,
                    Dir.right,
                ):
                    snake.direction = Dir.up

                elif event.key == pygame.K_DOWN and snake.direction in (
                    Dir.left,
                    Dir.right,
                ):
                    snake.direction = Dir.down

    @staticmethod
    def moveToList(move):
        tmpList = [0, 0, 0]
        tmpList[move] = 1
        return tmpList

    @staticmethod
    def directionToList(direction):
        tmpList = [0, 0, 0, 0]
        tmpList[direction] = 1
        return tmpList

    def getScore(self) -> int:
        if len(self.snakes) > 0:
            return self.snakes[0].length - self.snakes[0].initLength
        else:
            return 0

    def displayScore(self):
        self.displayInfo(
            f"Score: {self.score}", (SNAKE_GRID_SIZE_ABSOLUTE[0] - 80, SNAKE_GRID_SIZE_ABSOLUTE[1] - 20), 15
        )

    def displayInfo(self, info, position, size=22):
        font = pygame.font.SysFont("arial", size)
        text = font.render(info, True, (200, 200, 200), BG_COLOR)
        textRect = text.get_rect()
        textRect.center = position
        self.screen.blit(text, textRect)

    def draw(self, snakeReset=False, hudInfo=""):

        self.screen.fill(color=BG_COLOR)
        self.displayGrid(self.screen)

        for apple in self.apples:
            apple.draw()

        for snake in self.snakes:
            snake.move()
            snake.eat(self.apples)

            snake.draw()

            if snake.isDead():
                self.snakes.remove(snake)
                if snakeReset:
                    self.snakes.append(Snake(self.screen))

        if self.debugScreen:
            self.displayDebugScreen(self.screen)

        self.score = self.getScore()

        if hudInfo:
            self.displayInfo(
                hudInfo,
                (SNAKE_GRID_SIZE_ABSOLUTE[0] - 200, SNAKE_GRID_SIZE_ABSOLUTE[1] - 20),
                15,
            )
        self.displayScore()

        pygame.display.flip()

    def run(self):

        while True:
            self.moveOnEvent()

            if not self.paused:
                self.draw(snakeReset=True)

            pygame.time.wait(int(1000 / SPEED))


if __name__ == "__main__":
    snakeGame = SnakeGame(appleCount=1, snakeCount=1, displayDebugScreen=True)
    snakeGame.run()
