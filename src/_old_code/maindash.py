import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import run_core as rc

# --- Constants ---
REFRESH_RATE = 0.0  # Minimum interval between updates, in seconds

# --- Functions ---
def main():
    st.title("Cognitive Map Simulation")

    # Initialize the controls for running and restarting
    running = st.sidebar.radio("Control", ["Run", "Pause"], index=0)
    restart = st.sidebar.button("Restart Simulation")
    grid_placeholder = st.empty()  # Placeholder for the grid layout
    pause_message = st.sidebar.empty()  # Placeholder for "Paused" message

    # Initialize the simulation
    simulation = rc.Simulation(seed=0, plot_interval=10)

    # Handle restart button
    # Reset the simulation instance
    if restart:
        simulation = rc.Simulation(seed=0, plot_interval=10)
        st.sidebar.write("Simulation restarted!")

    # Time tracking to optimize update frequency
    last_update_time = time.time()

    # Continuous update loop
    while True:
        if running == "Run":
            # Update only if sufficient time has passed
            if time.time() - last_update_time > REFRESH_RATE:
                plot_list = simulation.update()  # Generate a new list of plots

                # Refresh the grid layout
                with grid_placeholder.container():
                    # Dynamically calculate grid layout based on the number of plots
                    num_plots = len(plot_list)
                    cols_per_row = 3  # Number of columns per row
                    cols = st.columns(cols_per_row)

                    # Loop through the plots and assign each to a column
                    for idx, fig in enumerate(plot_list):
                        # Adjust figure size for better visibility
                        fig.set_size_inches(5, 5)

                        col = cols[idx % cols_per_row]  # Cycle through the columns
                        with col:
                            st.pyplot(fig)

                last_update_time = time.time()  # Update the last update time
        else:
            pause_message.write("Paused")  # Display "Paused" only once
            time.sleep(0.1)  # Small delay to allow switching back to "Run"

if __name__ == "__main__":
    main()
