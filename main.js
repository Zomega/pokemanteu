import * as tf from 'https://esm.run/@tensorflow/tfjs';
import { MultiMarkovGenerator, MarkovLogitsProcessor } from './markov.js';
import {
    decodeBeamBatched,
    computeCyclicLoss,
    computeCyclicLossBatch,
    CyclicSequenceScorer,
} from './inference.js';

// --- 1. CONFIGURATION & GLOBAL STATE ---
const CONFIG = {
    MODEL_URL: "./tfjs_model/model.json",
    VOCAB_URL: "./vocab.json",
    MARKOV_URL: "./multi_markov.json",
    MARKOV_WEIGHTS: { "pokemon": 0.8, "english": 0.2 } // Extracted to config
};

// State object to keep things organized
const AppState = {
    model: null,
    vocab: null,
    invVocab: null,
    markovGenerator: null
};


// --- 2. INITIALIZATION ---
async function initializeApp() {
    try {
        console.log("Starting initialization...");

        // Fire off all network requests concurrently for faster loading
        const [model, vocabResp, markovResp] = await Promise.all([
            tf.loadGraphModel(CONFIG.MODEL_URL),
            fetch(CONFIG.VOCAB_URL),
            fetch(CONFIG.MARKOV_URL)
        ]);

        // Parse JSON responses
        const vocab = await vocabResp.json();
        const markovData = await markovResp.json();

        // Assign to global state
        AppState.model = model;
        AppState.vocab = vocab;
        AppState.invVocab = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));
        AppState.markovGenerator = new MultiMarkovGenerator(markovData);

        console.log("App ready! TFJS and Markov models loaded.");
    } catch (error) {
        console.error("Failed to initialize app:", error);
    }
}


// --- 3. BUSINESS LOGIC (Pure Pipeline, No DOM) ---
async function generateBidirectionalPronunciation(word) {
    if (!AppState.model) throw new Error("Model not loaded yet.");

    const { model, vocab, invVocab, markovGenerator } = AppState;

    // 1. Define Modifiers
    const logitsProcessors = [];
    if (markovGenerator) {
        logitsProcessors.push({
            processor: new MarkovLogitsProcessor(markovGenerator, CONFIG.MARKOV_WEIGHTS),
            weight: 0.2 // Lambda Markov
        });
    }

    const sequenceScorers = [
        {
            scorer: new CyclicSequenceScorer(model, vocab, invVocab, computeCyclicLossBatch),
            weight: 10 // Lambda Bidirectional Rerank
        }
    ];

    // 2. Run Forward Pass (Word -> IPA) with Markov steering
    const ipa = await decodeBeamBatched(
        model, vocab, invVocab, word, "<", 10, 1.0, 0.6,
        logitsProcessors, sequenceScorers
    );

    // 3. Run Backward Pass (IPA -> Word) without Markov steering (Markov only knows IPA!)
    const back_word = await decodeBeamBatched(
        model, vocab, invVocab, ipa, ">", 10, 1.0, 0.6,
        [], sequenceScorers
    );

    // 4. Calculate final diagnostic loss
    const cyclicData = await computeCyclicLoss(model, vocab, invVocab, ipa, word, ">");

    // Return the clean data object
    return { ipa, back_word, cyclicData };
}


// --- 4. UI HANDLERS (Reads DOM, Calls Logic, Updates DOM) ---

// Hook up the Inference Button
document.getElementById('inference-btn').addEventListener('click', async () => {
    const inputEl = document.getElementById("inputWord");
    const outputSpan = document.getElementById('output');
    const word = inputEl.value.trim();

    if (!word) return;

    // UI Loading State
    outputSpan.innerText = "Processing Bidirectional Beam Search...";
    inputEl.disabled = true;

    try {
        // Call the pure logic function
        const { ipa, back_word, cyclicData } = await generateBidirectionalPronunciation(word);

        // Map data for the console table
        const tableData = cyclicData.details.map((row, index) => ({
            Step: index + 1,
            Char: row.char,
            Prob: row.prob,
            Loss: row.loss,
        }));

        console.log(`%cCyclic Loss Analysis for "${word}"`, "font-weight: bold; font-size: 14px;");
        console.table(tableData);

        // Update DOM with results
        outputSpan.innerHTML = `
            <strong>IPA:</strong> ${ipa} <br>
            <strong>Reversed Word:</strong> ${back_word} <br>
            <strong>Avg Cyclic Loss:</strong> ${cyclicData.avgLoss} <br>
            <span style="font-size: 0.85em; color: gray;">
                (Lower is better. High loss on specific chars indicates phonetic ambiguity.)
            </span>
        `;
    } catch (err) {
        console.error(err);
        outputSpan.innerText = `Error: ${err.message}`;
    } finally {
        inputEl.disabled = false; // Release UI lock
    }
});

// Hook up the Markov Output Button
// Hook up the Markov Output Button
document.getElementById('generate-btn').addEventListener('click', async () => {
    // Ensure both models are loaded before running
    if (!AppState.markovGenerator || !AppState.model) {
        console.warn("Models not fully loaded yet.");
        return;
    }

    const outputDiv = document.getElementById('markov-output');
    outputDiv.innerHTML = "<em>Generating and decoding 10 words... Please wait.</em>";

    const { model, vocab, invVocab } = AppState;

    // We can reuse the CyclicSequenceScorer for the reverse translation
    const sequenceScorers = [{
        scorer: new CyclicSequenceScorer(model, vocab, invVocab, computeCyclicLossBatch),
        weight: 10
    }];

    // Start building the HTML table
    let tableHTML = `
        <table border="1" style="border-collapse: collapse; margin-top: 15px; text-align: left; width: 100%; max-width: 400px;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 8px;">Generated IPA</th>
                    <th style="padding: 8px;">Guessed Spelling</th>
                </tr>
            </thead>
            <tbody>
    `;

    // Generate 10 words
    for (let i = 0; i < 10; i++) {
        // 1. Generate the random IPA using the Markov model
        const ipa = AppState.markovGenerator.generate(CONFIG.MARKOV_WEIGHTS, 4, 12);

        // 2. Feed the IPA into the Neural Net to guess the spelling (Task Token ">")
        // Note: We pass [] for logitsProcessors because Markov only knows IPA, not English letters!
        const guessedWord = await decodeBeamBatched(
            model, vocab, invVocab, ipa, ">", 10, 1.0, 0.6,
            [], sequenceScorers
        );

        // Append the row to the table
        tableHTML += `
            <tr>
                <td style="font-family: monospace; padding: 8px;">/${ipa}/</td>
                <td style="padding: 8px; font-weight: bold; text-transform: capitalize;">${guessedWord}</td>
            </tr>
        `;
    }

    tableHTML += `</tbody></table>`;

    // Render the final table to the DOM
    outputDiv.innerHTML = tableHTML;
});

// Start the app
initializeApp();