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
    MARKOV_PHONEMES_URL: "./markov_phonemes.json",
    MARKOV_GRAPHEMES_URL: "./markov_graphemes.json"
};

const AppState = {
    model: null,
    vocab: null,
    invVocab: null,
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
        AppState.model = model;
        AppState.vocab = vocab;
        AppState.invVocab = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));

        AppState.phonemeMarkov = new MultiMarkovGenerator(phonemeData);
        AppState.graphemeMarkov = new MultiMarkovGenerator(graphemeData);

        console.log("App ready! TFJS and dual Markov models loaded.");
    } catch (error) {
        console.error("Failed to initialize app:", error);
    }
}


// --- 3. BUSINESS LOGIC (Pure Pipeline, No DOM) ---
async function generateBidirectionalPronunciation(word) {
    if (!AppState.model) throw new Error("Model not loaded yet.");

    const { model, vocab, invVocab, phonemeMarkov, graphemeMarkov } = AppState;

    // 1. Define Modifiers
    const logitsProcessors = [];

    // Add Phoneme Markov (Applies only when task is "<")
    if (phonemeMarkov) {
        logitsProcessors.push({
            processor: new MarkovLogitsProcessor("Markov_IPA", phonemeMarkov, null, "<"),
            weight: 0.2
        });
    }

    // Add Grapheme Markov (Applies only when task is ">")
    if (graphemeMarkov) {
        logitsProcessors.push({
            processor: new MarkovLogitsProcessor("Markov_Word", graphemeMarkov, null, ">"),
            weight: 0.4
        });
    }

    const sequenceScorers = [
        {
            scorer: new CyclicSequenceScorer(model, vocab, invVocab, computeCyclicLossBatch),
            weight: 10
        }
    ];

    // 2. Run Forward Pass (Word -> IPA)
    // The "Markov_IPA" processor will automatically engage because task is "<"
    const ipa = await decodeBeamBatched(
        model, vocab, invVocab, word, "<", 10, 1.0, 0.6,
        logitsProcessors, sequenceScorers
    );

    // 3. Run Backward Pass (IPA -> Word)
    // The "Markov_Word" processor will automatically engage because task is ">"
    const back_word = await decodeBeamBatched(
        model, vocab, invVocab, ipa, ">", 10, 1.0, 0.6,
        logitsProcessors, sequenceScorers
    );

    // 4. Calculate final diagnostic loss
    const cyclicData = await computeCyclicLoss(model, vocab, invVocab, ipa, word, ">");

    return { ipa, back_word, cyclicData };
}


// --- 4. UI HANDLERS (Reads DOM, Calls Logic, Updates DOM) ---

// Hook up the Inference Button
document.getElementById('inference-btn').addEventListener('click', async (event) => {
    // 1. Guard clause: Ensure TFJS is loaded
    if (!AppState.model) {
        alert("Models are still loading. Please wait a moment!");
        return;
    }

    const inputEl = document.getElementById("inputWord");
    const buttonEl = event.target; // The button that was clicked
    const outputSpan = document.getElementById('output');
    const word = inputEl.value.trim();

    if (!word) return;

    // 2. UI Loading State (Lock both input and button)
    outputSpan.innerHTML = "<em>Processing Bidirectional Beam Search...</em>";
    inputEl.disabled = true;
    buttonEl.disabled = true;

    try {
        // Call the pure logic function (which now handles BOTH Markov models internally)
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

        // 3. Update DOM with results
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
        // 4. Release UI locks
        inputEl.disabled = false;
        buttonEl.disabled = false;
        inputEl.focus(); // Nice UX touch: put cursor back in the box
    }
});

// Hook up the Markov Output Button
// Hook up the Markov Output Button
document.getElementById('generate-btn').addEventListener('click', async () => {
    if (!AppState.phonemeMarkov || !AppState.model) {
        console.warn("Models not fully loaded yet.");
        return;
    }

    const outputDiv = document.getElementById('markov-output');
    outputDiv.innerHTML = "<em>Generating and decoding 10 words... Please wait.</em>";

    const { model, vocab, invVocab, graphemeMarkov, phonemeMarkov } = AppState;

    // Use Grapheme Markov to guide the spelling generation
    const logitsProcessors = graphemeMarkov ? [{
        processor: new MarkovLogitsProcessor("Markov_Word", graphemeMarkov, null, ">"),
        weight: 0.2
    }] : [];

    const sequenceScorers = [{
        scorer: new CyclicSequenceScorer(model, vocab, invVocab, computeCyclicLossBatch),
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
        // Generate IPA. We don't pass weights; it defaults to the embedded JSON weights!
        const ipa = phonemeMarkov.generate(null, 4, 12);

        // Feed IPA into Neural Net to guess spelling, guided by the English/Poke spelling Markov model!
        const guessedWord = await decodeBeamBatched(
            model, vocab, invVocab, ipa, ">", 10, 1.0, 0.6,
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