import unittest
import geneticLearning


class TestGeneticLearning(unittest.TestCase):
    def test_crossOver(self):
        agent1 = geneticLearning.Agent()
        initWeights1 = agent1.getWeights()

        agent2 = geneticLearning.Agent()
        initWeights2 = agent2.getWeights()

        newWeights = agent1.crossOver(initWeights1, initWeights2)
        newWeightsFromWeights1 = False
        newWeightsFromWeights2 = False
        for i in range(len(newWeights[0][0])):
            if newWeights[0][0][i] == initWeights1[0][0][i]:
                newWeightsFromWeights1 = True
            if newWeights[0][0][i] == initWeights2[0][0][i]:
                newWeightsFromWeights2 = True

        self.assertTrue(newWeightsFromWeights1)
        self.assertTrue(newWeightsFromWeights2)

    def test_mutate_weights(self):

        agent = geneticLearning.Agent()
        initialWeights = agent.getWeights()

        rate = 0.1
        newWeights = agent.mutateWeights(initialWeights, rate=rate)

        mutated = 0
        totalWeights = 0
        for i in range(len(initialWeights)):
            if len(initialWeights[i].shape) != 2:
                for k in range(len(initialWeights[i])):
                    totalWeights += 1
                    if initialWeights[i][k] != newWeights[i][k]:
                        mutated += 1
                continue

            for j in range(len(initialWeights[i])):
                for k in range(len(initialWeights[i][j])):

                    totalWeights += 1

                    if initialWeights[i][j][k] != newWeights[i][j][k]:
                        mutated += 1

        self.assertGreater(mutated / totalWeights, rate - 0.05)
        self.assertLess(mutated / totalWeights, rate + 0.05)


if __name__ == "__main__":
    unittest.main()
