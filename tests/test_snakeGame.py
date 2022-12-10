import os
import unittest
import snakeGame


# run in headless mode for the tests
os.environ["SDL_VIDEODRIVER"] = "dummy"


class TestSnakeGame(unittest.TestCase):
    def test_apple_relative_position(self):
        game = snakeGame.SnakeGame(appleCount=1, snakeCount=1)
        snake = snakeGame.Snake(screen=game.screen)
        snake.headPosition = (10, 10)
        apple1 = snakeGame.Apple(game)
        apple1.position = (0, 0)

        apple2 = snakeGame.Apple(game)
        apple2.position = (5, 5)

        apple3 = snakeGame.Apple(game)
        apple3.position = (12, 13)

        game.apples = [apple1, apple2, apple3]
        game.snakes = [snake]

        self.assertEqual(snake.getNearestApple(game.apples), apple3)

    def test_get_danger_grid_when_snake_is_in_the_middle(self):
        game = snakeGame.SnakeGame(appleCount=1, snakeCount=1)
        snake = snakeGame.Snake(screen=game.screen)
        snake.headPosition = (10, 10)
        snake.bodyQueue = [(10, 10), (10, 11), (9, 11), (9, 10), (8, 10), (7, 10)]

        snake.direction = snakeGame.Dir.up

        game.snakes = [snake]

        expectedGrid = [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

        for a, b in zip(snake.getDangerGrid(), expectedGrid):
            self.assertListEqual(list(a), list(b))

    def test_get_danger_grid_when_snake_is_touching_the_bounds(self):
        game = snakeGame.SnakeGame(appleCount=1, snakeCount=1)
        snake = snakeGame.Snake(screen=game.screen)
        snake.headPosition = (1, 1)
        snake.bodyQueue = [snake.headPosition, (1, 1), (2, 1)]

        snake.direction = snakeGame.Dir.right

        game.snakes = [snake]
        expectedGrid = [[0, 0, 0], [1, 1, 0], [0, 0, 0]]

        for a, b in zip(snake.getDangerGrid(), expectedGrid):
            self.assertListEqual(list(a), list(b))


if __name__ == "__main__":
    unittest.main()
