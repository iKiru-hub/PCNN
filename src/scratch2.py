import run_core as rc
import matplotlib.pyplot as plt



if __name__ == '__main__':

    sim = rc.Simulation()

    figs = sim.update()
    figs = sim.update()

    print(figs)
    plt.show()

    # plot the figures
    for fig in figs:
        fig.show()
        # st.pyplot(fig)
        # plt.show()
        # plt.close(fig)
