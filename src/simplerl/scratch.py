import numpy as np
# convolve 1d
from numpy import convolve

from scipy.stats import beta, dirichlet
import matplotlib.pyplot as plt
import argparse
import tqdm

from stable_baselines3 import PPO, A2C, TD3, DDPG, DQN


class Env:
    def __init__(self):
        self.curr = 1
    def __call__(self, x, y, t):
        # t = t / 10000 + 1
        # self.curr = self.curr if t % 500 != 0 else np.random.choice([-1, 0, 1])

        # return 0.5*(np.sin(2*x) * 1*(y==self.curr) + np.cos(x))

        if t % 10000 == 0:
            self.curr *= -1
        
        if self.curr == 1:
            result = np.where(x > 0.8,
                            np.where(y > 0.5, 1, -1),
                            -1)
            return result
        result = np.where(x < 0.3,
                            np.where(y <-0.5, 1, -1),
                            -1)
        return result
        # return 0.5*(np.sin(4*x+1) * np.cos(2*y+t) + np.cos(2*x+t))

    def plot(self, ax: plt.Axes=None, alpha=0.5, t=0):

        show = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            show = True

        # Generate a meshgrid for plotting
        x = np.linspace(-0.2, 1.2, 400)
        y = np.linspace(-1.2, 1.2, 400)
        X, Y = np.meshgrid(x, y)
        Z = self(X, Y, t)

        # Plot the function
        ax.axhline(1, color='black', lw=0.5, alpha=0.3)
        ax.axhline(0, color='black', lw=0.5, alpha=0.3)
        ax.axhline(-1, color='black', lw=0.5, alpha=0.3)
        ax.axvline(0, color='black', lw=0.5, alpha=0.3)
        ax.axvline(1, color='black', lw=0.5, alpha=0.3)

        cp = ax.contourf(X, Y, Z, cmap='Greys',
                         alpha=alpha, vmin=-1, vmax=1)
        # ax.imshow(Z, cmap='Greens', alpha=alpha)
        ax.set_title(f'Environment, curr={self.curr}')
        ax.set_xlabel('$\\epsilon$')
        ax.set_ylabel('$\\lambda$')
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        # ax.grid(True)

        if show:
            plt.show()



class BayesianPolicy2:

    """
    Bayesian policy to choose between two actions:
    - epsilon: the weight for the representation sum
    - lambda: the bias in the drift representation
    """

    def __init__(self, alpha_beta=1, beta_beta=1,
                 alpha_dirichlet=np.array([1, 1, 1]),
                 lr: float=0.1, method: str="sample"):

        # Initialize priors
        self.alpha_beta = alpha_beta
        self.beta_beta = beta_beta
        self.alpha_dirichlet = alpha_dirichlet
        self.lr = lr
        assert method in ["sample", "expected", "random"], \
            "Method must be 'sample' or 'expected'"
        self.method = method

        self.last_action = None

    def __call__(self):

        if self.method == "sample":
            # Sample from Beta distribution for first component
            action1 = beta.rvs(self.alpha_beta,
                               self.beta_beta) #> 0.5

            # Sample from Dirichlet distribution for second component
            action2_probs = dirichlet.rvs(self.alpha_dirichlet)[0]
            action2 = np.random.choice(3, p=action2_probs)
        elif self.method == "expected":
            # Get expected values
            action1, action2 = self.get_expected_values()

        elif self.method == "random":
            action1 = np.random.rand()
            action2 = np.random.choice(3)

        self.last_action = (action1, action2)

        return action1, action2

    def update(self, reward: int):

        action1, action2 = self.last_action

        if reward == 0:
            return

        update_value = self.lr * reward

        if reward > 0:
            if action1:
                self.alpha_beta += update_value
            else:
                self.beta_beta += update_value
            self.alpha_dirichlet[action2] += update_value
        else:
            if not action1:
                self.beta_beta -= update_value
                self.alpha_beta += update_value / 2
            else:
                self.alpha_beta -= update_value
                self.beta_beta += update_value / 2

        min_p = 0.1
        # self.alpha_dirichlet = np.clip(self.alpha_dirichlet, min_p, max_p)
        self.alpha_dirichlet = np.maximum(self.alpha_dirichlet, min_p)
        self.alpha_beta = np.clip(self.alpha_beta, min_p, 1.)
        self.beta_beta = np.clip(self.beta_beta, min_p, 1.)
        # self.alpha_beta = max(self.alpha_beta, min_p)
        # self.beta_beta = max(self.beta_beta, min_p)

    def get_expected_values(self):
        # ev_action1 = self.alpha_beta / (self.alpha_beta + self.beta_beta)
        ev_action1 = self.beta_beta / (self.alpha_beta + self.beta_beta)
        ev_action2 = self.alpha_dirichlet / np.sum(self.alpha_dirichlet)
        return ev_action1, ev_action2

class BayesianPolicy:
    def __init__(self, alpha_beta=1, beta_beta=1,
                 alpha_dirichlet=np.array([1, 1, 1]),
                 lr=0.1, method: str="sample"):
        self.alpha_beta = alpha_beta
        self.beta_beta = beta_beta
        self.alpha_dirichlet = alpha_dirichlet
        self.learning_rate = lr
        self.method = method
        self.last_action = None

    def __call__(self):
        if self.method == "sample":
            action1 = beta.rvs(self.alpha_beta, self.beta_beta) > 0.5
            action2_probs = dirichlet.rvs(self.alpha_dirichlet)[0]
            action2 = np.random.choice(3, p=action2_probs)
        elif self.method == "random":
            action1 = np.random.rand() > 0.5
            action2 = np.random.choice(3)

        self.last_action = (action1, action2)
        return action1, action2

    def update(self, reward):
        action1, action2 = self.last_action
        
        if reward == 0:
            return

        update_value = self.learning_rate * reward
        
        if reward > 0:
            if action1:
                self.alpha_beta += update_value
            else:
                self.beta_beta += update_value
            self.alpha_dirichlet[action2] += update_value
        else:
            if action1:
                self.beta_beta -= update_value
                self.alpha_beta += update_value / 2
            else:
                self.alpha_beta -= update_value
                self.beta_beta += update_value / 2
            
            penalty = update_value / (len(self.alpha_dirichlet) - 1)
            for i in range(len(self.alpha_dirichlet)):
                if i == action2:
                    self.alpha_dirichlet[i] -= penalty
                else:
                    self.alpha_dirichlet[i] += penalty
            
        self.alpha_dirichlet = np.maximum(self.alpha_dirichlet, 0.1)
        self.alpha_beta = max(self.alpha_beta, 0.01)
        self.beta_beta = max(self.beta_beta, 0.01)

    def get_expected_values(self):
        ev_action1 = self.alpha_beta / (self.alpha_beta + self.beta_beta)
        ev_action2 = self.alpha_dirichlet / np.sum(self.alpha_dirichlet)
        return ev_action1, ev_action2


class BetaBayes:

    def __init__(self, lr=0.1, min_value=0.1):

        self.alpha = 1
        self.beta = 1
        self.lr = lr
        self.action = 0
        self.min_value = min_value

    def __call__(self):
        self.action = beta.rvs(self.alpha, self.beta)
        return 1*(self.action>0.5)

    def update(self, reward):
 
        if reward == 0:
            return

        update_value = self.lr * reward

        if self.action > 0.5:
            self.beta *= 1 - update_value
        else:
            self.beta *= 1 + update_value

        # normalization
        self.beta = np.clip(self.beta, self.min_value,
                            1/self.min_value)


class DirichletBayes:

    def __init__(self, lr=0.1, min_value=0.1, K: int=3):

        self.alphas = np.ones(K)
        self.K = K
        self.lr = lr
        self.action = 0
        self.min_value = min_value

    def __call__(self):
        self.action = np.random.choice(self.K,
                                       p=dirichlet.rvs(self.alphas)[0])
        return self.action

    def update(self, reward):
 
        if reward == 0:
            return

        update_value = self.lr * reward

        for i in range(self.K):
            if i == self.action:
                self.alphas[i] *= 1 + update_value
            else:
                self.alphas[i] *= 1 - update_value

        self.alphas = np.clip(self.alphas, self.min_value,
                                    1/self.min_value)



def run(env, agent, agent2, duration=1000, tplot=10):

    rewards = np.zeros((duration, 3))
    rewards2 = np.zeros((duration, 3))

    actions = np.zeros((duration, 2))

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs = axs.flatten()

    for t in range(duration):

        action1, action2 = agent()
        action2 = action2 -1

        actions[t] = [action1, action2]
        # action1 = round(action1, 3)

        action1_2, action2_2 = agent2()
        action2_2 = action2_2 - 1

        reward = env(action1, action2, t=t)
        reward2 = env(action1_2, action2_2, t=t)

        agent.update(reward)

        rewards[t] = [action1, action2, reward]
        rewards2[t] = [action1_2, action2_2, reward2]

        # plot
        if t % tplot == 0:

            axs[0].clear()
            axs[1].clear()
            axs[2].clear()
            axs[3].clear()

            fig.suptitle(f'Time: {t}')

            # env
            env.plot(ax=axs[0], t=t)

            # rewards
            tr = 100
            axs[0].scatter(rewards[max((0, t-tr)):t, 0],
                           rewards[max((0, t-tr)):t, 1],
                           c=rewards[max((0, t-tr)):t, 2],
                           cmap='RdYlGn', s=40, alpha=0.9,
                           vmin=-1, vmax=1)

            # axs[0].scatter(rewards2[max((0, t-tr)):t, 0],
            #                rewards2[max((0, t-tr)):t, 1],
            #                c=rewards2[max((0, t-tr)):t, 2],
            #                cmap='RdYlGn', s=40, alpha=0.5,
            #                marker='x', vmin=-1, vmax=1)

            tr = 100
            axs[3].bar(np.arange(3), agent.alpha_dirichlet/np.sum(agent.alpha_dirichlet),
                       alpha=0.8)
            axs[3].set_title('Dirichlet')
            axs[3].set_xticks([0, 1, 2])
            axs[3].set_ylim(0, 1)

            # normalize beta values
            axs[2].bar([0, 1], [agent.alpha_beta/(agent.alpha_beta + agent.beta_beta),
                                agent.beta_beta/(agent.alpha_beta + agent.beta_beta)],
                       alpha=0.8)
            axs[2].set_title('Beta')
            axs[2].set_xticks([0, 1])
            axs[2].set_ylim(0, 1)

            avg_action = actions[max((0, t-tr)):t].mean(axis=0)
            ea1, ea2 = agent.get_expected_values()

            print(f"\nTime: {t}, Reward: {reward:.2f}, " + \
                    f"Expected: {ea1:.2f}, {ea2[0]:.2f}, {ea2[1]:.2f}, {ea2[2]:.2f} ")
            print(f"Action: {action1:.2f}, {action2:.2f}")


            # rew_mean = rewards[max((0, t-tr)):t, 2].mean()
            # smooth the rewards
            if t > 10:
                rew_mean = convolve(rewards[:t, 2], np.ones(10)/10, mode='valid')
                rew_mean2 = convolve(rewards2[:t, 2], np.ones(10)/10, mode='valid')
            else:
                rew_mean = rewards[:, 2]
                rew_mean2 = rewards2[:, 2]

            axs[1].plot(rew_mean,
                        label=f"sample, r={rewards[max((0, t-tr)):t, 2].mean():.2f}",
                        alpha=0.5)
            axs[1].plot(rew_mean2,
                        label=f"random, {rewards2[max((0, t-tr)):t, 2].mean():.2f}",
                        alpha=0.5)
            axs[1].set_title(f'Rewards')
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('R')
            axs[1].set_ylim(-1.2, 1.2)
            axs[1].set_xlim((max((t-1000, 0)),
                            t))
            axs[1].legend(loc="lower right")

            plt.pause(0.001)

    plt.show()


if __name__ == "__main__":

    env = Env()
    agent = BayesianPolicy(lr=.9,
                           method="sample")
    agent2 = BayesianPolicy(lr=.2,
                              method="random")

    run(env, agent, agent2, duration=100_000, tplot=40)

