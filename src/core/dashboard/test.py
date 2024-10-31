import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


PATH = "media/"


class DummyPlot:

    def __init__(self, name: str):

        self.name = str(name)
        self.id = "".join([str(np.random.randint(0, 9)) for _ in range(10)])
        self.fig, self.ax = plt.subplots()

        self.freqs = np.random.uniform(-10, 10, 2).tolist()

        self.t = 0.
        self.data = []

    def _make_data(self):

        self.t += 0.1

        self.data += [np.sin(self.t*self.freqs[0]) + \
                      0.5*np.random.randn() + \
                      self.freqs[1]*np.sin(self.t*0.001)]

        if len(self.data) > 200:
            del self.data[0]

        return self.data

    def _save(self):

        self.fig.savefig(f"{PATH}fig{self.name}.png")

    def __call__(self, t: int):

        if t % 1000 == 0:

            self.ax.clear()
            self.ax.plot(self._make_data())
            self.ax.set_title(f"Plot {self.name} - {self.id}")
            self.ax.set_ylim(-3, 3)
            self._save()


def simulation(args):

    num_figs = args.N
    plots = [DummyPlot(i) for i in range(num_figs)]

    info = {
        "num_figs": num_figs
    }

    with open(f"{PATH}configs.json", "w") as f:
        json.dump(info, f)

    if args.duration == 0:
        t = 0
        while True:
            for plot in plots:
                plot(t=t)
            t += 1

    else:
        for t in range(args.duration):
            print(f"{t=}")
            for plot in plots:
                plot(t=t)

    print("----\ndone")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--duration", type=int, default=100)
    args = parser.parse_args()

    simulation(args)

