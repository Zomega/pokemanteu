// 1. Import your dependencies matching the actual exports
import { MultiMarkovGenerator } from './markov.js';
import { loadResources, runInference } from './inference.js';

// 2. Global application state
let markovGenerator = null;

// 3. Initialize everything when the page loads
async function initializeApp() {
    try {
        console.log("Starting initialization...");

        // Load TFJS Model & Vocab (handled entirely by inference.js)
        await loadResources();

        // Load Markov JSON
        const markovResponse = await fetch('multi_markov.json');
        const markovData = await markovResponse.json();
        markovGenerator = new MultiMarkovGenerator(markovData);
        console.log("Markov model ready.");

    } catch (error) {
        console.error("Failed to initialize app:", error);
    }
}

// 4. Wire up the DOM elements

// Hook up the Inference Button
document.getElementById('inference-btn').addEventListener('click', async () => {
    const outputSpan = document.getElementById('output');
    outputSpan.innerText = "Processing Bidirectional Beam Search...";

    // runInference() handles reading the input and updating the DOM automatically
    await runInference();
});

// Hook up the Markov Button
document.getElementById('generate-btn').addEventListener('click', () => {
    if (!markovGenerator) {
        console.warn("Markov generator not loaded yet.");
        return;
    }

    const outputDiv = document.getElementById('markov-output');
    outputDiv.innerHTML = "";

    // Tweak these weights to change the dialect!
    const weights = { "pokemon": 0.8, "english": 0.2 };

    for (let i = 0; i < 10; i++) {
        const word = markovGenerator.generate(weights, 4, 12);
        outputDiv.innerHTML += `<div style="font-family: monospace; padding: 4px;">/${word}/</div>`;
    }
});

// Start the app
initializeApp();