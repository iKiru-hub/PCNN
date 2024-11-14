import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import run_core as rc

# --- Constants ---
REFRESH_RATE = 0.0001  # Refresh rate in seconds

# --- Functions ---

def main():

    st.title("Cognitive Map Simulation")

    # Initialize the controls for running and restarting
    running = st.sidebar.radio("Control", ["Run", "Pause"], index=0)
    restart = st.sidebar.button("Restart Simulation")
    grid_placeholder = st.empty()  # Placeholder for the grid layout
    pause_message = st.sidebar.empty()  # Placeholder for "Paused" message

    # Initialize the simulation
    simulation = rc.Simulation(seed=0,
                               plot_interval=10)

    # Handle restart button
    if restart:
        simulation = rc.Simulation()  # Reset the simulation instance
        st.sidebar.write("Simulation restarted!")

    # Continuous update loop
    while True:
        if running == "Run":
            plot_list = simulation.update()  # Generate a new list of plots

            # Define the number of columns for the grid and their width ratios
            cols = grid_placeholder.columns([2, 2, 2])  # Make each column wider (3 equal wide columns)

            # Loop through the plots and assign each to a column
            for idx, fig in enumerate(plot_list):

                # Display plot in a specific column in the grid
                col = cols[idx % 3]  # Cycle through the columns (3 per row)
                with col:
                    st.pyplot(fig)

            time.sleep(REFRESH_RATE)  # Adjust the refresh rate as needed
        else:
            pause_message.write("Paused")  # Display "Paused" only once
            time.sleep(0.1)  # Small delay to allow switching back to "Run"

if __name__ == "__main__":
    main()
