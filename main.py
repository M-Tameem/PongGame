import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import main
from pong.envs.utils import plotLearning

GAMMA = 0.99  # Weighting of future rewards
EPSILON = 1.0  # How long agents spends time exploring env vs taking best known actionone
LR = 0.003
BATCH_SIZE = 64
N_ACTIONS = 4
N_GAMES = 500
print("0")
if __name__ == "__main__":
    print("0.5")
    env = gym.make("Snake-v0")
    # f = SnakeEnv()
    print("1")
    scoreone = 0
    scoretwo = 0
    print("2")

    agentone = Agent(
        gamma=GAMMA,
        epsilon=EPSILON,
        batch_size=BATCH_SIZE,
        n_actions=N_ACTIONS,
        eps_end=0.01,
        input_dims=[3],
        lr=LR,
    )

    agenttwo = Agent(
        gamma=GAMMA,
        epsilon=EPSILON,
        batch_size=BATCH_SIZE,
        n_actions=N_ACTIONS,
        eps_end=0.01,
        input_dims=[3],
        lr=LR,
    )
    print("3")

    scoresone, eps_historyone = [], []
    scorestwo, eps_historytwo = [], []

    print("4")
    for i in range(N_GAMES):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            actionone = agentone.choose_action(observation)
            actiontwo = agenttwo.choose_action(observation)
            print("actionone: ", actionone)
            print("actiontwo: ", actiontwo)
            observation_, reward, done, info = env.step(actionone)
            observation_, reward, done, info = env.step(actiontwo)
            print("done 2")
            scoreone += reward
            scoretwo += -abs(reward)
            agentone.store_transition(observation, actionone, reward, observation_, done)
            agenttwo.store_transition(observation, actionone, reward, observation_, done)
            agentone.learn()
            agenttwo.learn()
            observation = observation_
        scoresone.append(scoreone)
        scorestwo.append(scoretwo)
        eps_historyone.append(agentone.epsilon)
        eps_historytwo.append(agenttwo.epsilon)

        avg_one_score = np.mean(scoresone[-100:])
        avg_two_score = np.mean(scoretwo[-100:])

        print(
            "episode: ",
            i,
            " Score: %.2f " % score,
            " Avg one score: %.2f" % avg_one_score,
            " Avg two score: %.2f" % avg_two_score,
            " One Epsilon: %.2f" % agentone.epsilon,
            " Two Epsilon: %.2f" % agentone.epsilon,
        )
    x = [i + 1 for i in range(N_GAMES)]
    filename = "pong_2022"
    plotLearning(x, scoresone, eps_historyone, filename)
    plotLearning(x, scorestwo, eps_historytwo, filename)