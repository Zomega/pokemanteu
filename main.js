import * as tf from 'https://esm.run/@tensorflow/tfjs';
import {
    MultiMarkovGenerator,
    MarkovLogitsProcessor
} from './markov.js';
import {
    InferenceEngine,
    CyclicSequenceScorer
} from './inference.js';

// --- 1. CONFIGURATION & GLOBAL STATE ---
const CONFIG = {
    MODEL_URL: "./tfjs_model/model.json",
    VOCAB_URL: "./vocab.json",
    MARKOV_PHONEMES_URL: "./markov_phonemes.json",
    MARKOV_GRAPHEMES_URL: "./markov_graphemes.json"
};

// We only need the engine and the two generators now!
const AppState = {
    engine: null,
    phonemeMarkov: null,
    graphemeMarkov: null
};


// --- 2. INITIALIZATION ---
async function initializeApp() {
    try {
        console.log("Starting initialization...");

        // Fire off all network requests concurrently
        const [model, vocabResp, phonemeResp, graphemeResp] = await Promise.all([
            tf.loadGraphModel(CONFIG.MODEL_URL),
            fetch(CONFIG.VOCAB_URL),
            fetch(CONFIG.MARKOV_PHONEMES_URL),
            fetch(CONFIG.MARKOV_GRAPHEMES_URL)
        ]);

        // Parse JSON
        const vocab = await vocabResp.json();
        const phonemeData = await phonemeResp.json();
        const graphemeData = await graphemeResp.json();

        // Assign to global state
        AppState.engine = new InferenceEngine(model, vocab, 40);
        AppState.phonemeMarkov = new MultiMarkovGenerator(phonemeData);
        AppState.graphemeMarkov = new MultiMarkovGenerator(graphemeData);

        console.log("App ready! TFJS and dual Markov models loaded.");
    } catch (error) {
        console.error("Failed to initialize app:", error);
    }
}


// --- 3. BUSINESS LOGIC (Pure Pipeline, No DOM) ---
async function generateBidirectionalPronunciation(word) {
    if (!AppState.engine) throw new Error("Engine not loaded yet.");

    const {
        engine,
        phonemeMarkov,
        graphemeMarkov
    } = AppState;

    const logitsProcessors = [];
    logitsProcessors.push({
        processor: new MarkovLogitsProcessor("Markov_IPA", phonemeMarkov, null, "<"),
        weight: 0.2
    });
    logitsProcessors.push({
        processor: new MarkovLogitsProcessor("Markov_Word", graphemeMarkov, null, ">"),
        weight: 0.2
    });

    const sequenceScorers = [{
        scorer: new CyclicSequenceScorer(engine),
        weight: 10
    }];

    const ipa = await engine.decodeBeamBatched(
        word, "<", 10, 1.0, 0.6,
        logitsProcessors, sequenceScorers
    );

    const back_word = await engine.decodeBeamBatched(
        ipa, ">", 10, 1.0, 0.6,
        logitsProcessors, sequenceScorers
    );

    const cyclicData = await engine.computeCyclicLoss(ipa, word, ">");

    return {
        ipa,
        back_word,
        cyclicData
    };
}


// --- 4. UI HANDLERS (Reads DOM, Calls Logic, Updates DOM) ---

// Hook up the Inference Button
document.getElementById('inference-btn').addEventListener('click', async (event) => {
    // FIX: Check AppState.engine instead of AppState.model
    if (!AppState.engine) {
        alert("Models are still loading. Please wait a moment!");
        return;
    }

    const inputEl = document.getElementById("inputWord");
    const buttonEl = event.target;
    const outputSpan = document.getElementById('output');
    const word = inputEl.value.trim();

    if (!word) return;

    outputSpan.innerHTML = "<em>Processing Bidirectional Beam Search...</em>";
    inputEl.disabled = true;
    buttonEl.disabled = true;

    try {
        const {
            ipa,
            back_word,
            cyclicData
        } = await generateBidirectionalPronunciation(word);

        const tableData = cyclicData.details.map((row, index) => ({
            Step: index + 1,
            Char: row.char,
            Prob: row.prob,
            Loss: row.loss,
        }));

        console.log(`%cCyclic Loss Analysis for "${word}"`, "font-weight: bold; font-size: 14px;");
        console.table(tableData);

        outputSpan.innerHTML = `
            <strong>IPA:</strong> /${ipa}/ <br>
            <strong>Reversed Word:</strong> ${back_word} <br>
            <strong>Avg Cyclic Loss:</strong> ${cyclicData.avgLoss} <br>
            <span style="font-size: 0.85em; color: gray;">
                (Lower is better. High loss on specific chars indicates phonetic ambiguity.)
            </span>
        `;
    } catch (err) {
        console.error(err);
        outputSpan.innerHTML = `<span style="color: red;">Error: ${err.message}</span>`;
    } finally {
        inputEl.disabled = false;
        buttonEl.disabled = false;
        inputEl.focus();
    }
});

// Hook up the Markov Output Button
document.getElementById('generate-btn').addEventListener('click', async () => {
    // FIX: Check AppState.engine instead of AppState.model
    if (!AppState.phonemeMarkov || !AppState.engine) {
        console.warn("Models not fully loaded yet.");
        return;
    }

    const outputDiv = document.getElementById('markov-output');
    outputDiv.innerHTML = "<em>Generating and decoding 10 words... Please wait.</em>";

    const {
        engine,
        graphemeMarkov,
        phonemeMarkov
    } = AppState;

    const logitsProcessors = graphemeMarkov ? [{
        processor: new MarkovLogitsProcessor("Markov_Word", graphemeMarkov, null, ">"),
        weight: 0.2
    }] : [];

    const sequenceScorers = [{
        scorer: new CyclicSequenceScorer(engine),
        weight: 10
    }];

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

    for (let i = 0; i < 10; i++) {
        const ipa = phonemeMarkov.generate(null, 4, 12);

        const guessedWord = await engine.decodeBeamBatched(
            ipa, ">", 3, 1.0, 0.6,
            logitsProcessors, sequenceScorers
        );

        tableHTML += `
            <tr>
                <td style="font-family: monospace; padding: 8px;">/${ipa}/</td>
                <td style="padding: 8px; font-weight: bold; text-transform: capitalize;">${guessedWord}</td>
            </tr>
        `;
    }

    tableHTML += `</tbody></table>`;
    outputDiv.innerHTML = tableHTML;
});

// Start the app
initializeApp();