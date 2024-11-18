// Paths and variables
const IMAGE_FOLDER = 'cache';
const CONFIG_PATH = 'cache/configs.json';
let imageCount = 4;
let autoUpdate = false;
let updateInterval = null;
const MAX_RETRIES = 1; // Maximum number of retries for loading each image
const REFRESH_INTERVAL = 700; // Interval for auto-refresh in milliseconds
let loadedImages = []; // Array to hold successfully loaded images

async function loadImages() {
    console.log("Loading images...");

    const grid = document.getElementById('grid');
    grid.innerHTML = ''; // Clear current images
    loadedImages = []; // Reset loaded images on each load

    for (let i = 0; i < imageCount; i++) {
        await loadImageWithRetry(i, MAX_RETRIES, grid);
    }

    console.log("Images loading initiated");

    // print the number of images loaded
    console.log(`Number of images loaded: ${grid.childElementCount}`);
}

async function loadImageWithRetry(index, retries, grid) {
    const img = document.createElement('img');
    img.src = `${IMAGE_FOLDER}/fig${index}.png?${new Date().getTime()}`; // Cache-busting
    img.alt = `Plot ${index}`;
    img.classList.add('plot-image');

    let attempt = 0;

    const loadImage = () => {
        return new Promise((resolve) => {
            img.onload = () => {
                console.log(`Image plot${index}.png loaded successfully`);
                grid.appendChild(img); // Append only if loading is successful
                loadedImages[index - 1] = img; // Store the successfully loaded image
                resolve();
            };

            img.onerror = () => {
                console.error(`Image plot${index}.png not found on attempt ${attempt + 1}`);
                if (attempt < retries) {
                    attempt++;
                    console.log(`Retrying image plot${index}.png... (Attempt ${attempt})`);
                    setTimeout(loadImage, REFRESH_INTERVAL); // Retry after 500ms
                } else {
                    console.warn(`Failed to load image plot${index}.png after ${retries} attempts`);
                    // Check if there was a previous successfully loaded image
                    if (loadedImages[index - 1]) {
                        console.log(`Retaining previous image for plot ${index}`);
                        grid.appendChild(loadedImages[index - 1].cloneNode()); // Re-append previous image
                    }
                    resolve(); // Resolve even if failed after retries
                }
            };
        });
    };

    await loadImage(); // Wait for the load image promise to resolve
}

async function updateImageCount() {
    console.log("Updating image count from configs.json...");
    try {
        const response = await fetch(CONFIG_PATH);
        const config = await response.json();
        imageCount = config.num_figs || 0;
        console.log(`Image count read: ${config.num_figs}`);
        console.log(`Image count set to: ${imageCount}`);
        loadImages(); // Load images with the updated image count
    } catch (error) {
        console.error("Could not load configs.json", error);
    }
}

function refreshImages() {
    console.log("Manual refresh triggered");
    updateImageCount();
    const refreshButton = document.getElementById('refreshButton');
    refreshButton.classList.add('active');
    setTimeout(() => refreshButton.classList.remove('active'), 300);
}

function toggleAutoUpdate() {
    autoUpdate = !autoUpdate;
    const toggleButton = document.getElementById('toggleUpdateButton');

    if (autoUpdate) {
        console.log("Auto-refresh enabled");
        toggleButton.classList.add('active');
        updateInterval = setInterval(updateImageCount, REFRESH_INTERVAL);
    } else {
        console.log("Auto-refresh disabled");
        toggleButton.classList.remove('active');
        clearInterval(updateInterval);
    }
}

window.onload = () => {
    console.log("Page loaded, initializing...");
    // updateImageCount();
    document.getElementById('refreshButton').onclick = refreshImages;
    document.getElementById('toggleUpdateButton').onclick = toggleAutoUpdate;
};


// Function to fetch and display logs from logs.json
async function fetchLogs() {
    try {
        const response = await fetch('cache/configs.json'); // Path to logs.json
        const logs = await response.json();
        const logContainer = document.getElementById('log-container');
        logContainer.innerHTML = ''; // Clear old logs

        // Populate log data
        for (const [key, value] of Object.entries(logs)) {
            const logItem = document.createElement('p');
            logItem.textContent = `${key}: ${value}`;
            logContainer.appendChild(logItem);
        }
    } catch (error) {
        console.error("Error loading logs:", error);
        document.getElementById('log-container').textContent = "Error loading logs";
    }
}

// Function to initialize the plot
function initializePlot() {
    // Plotly.newPlot('plot', [{ y: [], type: 'line' }]);

    setInterval(async () => {
        // const newValue = Math.random(); // Replace with your data source
        // Plotly.extendTraces('plot', { y: [[newValue]] }, [0]);

        // Refresh logs every 10 seconds
        fetchLogs();
    }, 300);
}

// Load logs and initialize plot on page load
fetchLogs();
initializePlot();

