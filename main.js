import * as tf from 'https://esm.run/@tensorflow/tfjs';
import { MultiMarkovGenerator } from './markov.js';
import { decodeBeamBatched, computeCyclicLoss, computeCyclicLossBatch, CyclicSequenceScorer } from './inference.js';

// 2. Global application state

// --- CONFIGURATION ---
// In the browser, these are URLs relative to the HTML file
const MODEL_URL = "./tfjs_model/model.json";
const VOCAB_URL = "./vocab.json";

let model = null;
let vocab = null;
let invVocab = null;
let markovGenerator = null;

// --- SETUP ---
export async function loadResources() {
  if (model) return;

  console.log("Loading model...");
  model = await tf.loadGraphModel(MODEL_URL);

  console.log("Loading vocab...");
  const vocabResp = await fetch(VOCAB_URL);
  vocab = await vocabResp.json();
  invVocab = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));

  console.log("Ready!");
}

export async function runInference() {
  await loadResources();
  const word = document.getElementById("inputWord").value.trim();

  // 1. Define your pipelines
  const logitsProcessors = [];
  // TODO: Hook up markov stuff properly.
  const markovGenerator  = null;
  if (markovGenerator) {
    logitsProcessors.push({
      processor: new MarkovLogitsProcessor(markovGenerator, markovWeights),
      weight: 0.2 // Lambda Markov
    });
  }

  const sequenceScorers = [
    {
      scorer: new CyclicSequenceScorer(model, vocab, invVocab, computeCyclicLossBatch),
      weight: 10 // Lambda Bidirectional Rerank
    }
  ];

  // 2. Run Inference
  // 1. Get the IPA via Beam Search
  const ipa = await decodeBeamBatched(model, vocab, invVocab, word, "<", 10, 1.0, 0.6, logitsProcessors, sequenceScorers);
  const back_word = await decodeBeamBatched(model, vocab, invVocab, ipa, ">", 10, 1.0, 0.6, [], sequenceScorers);

  // 2. Calculate the reverse forced loss
  const cyclicData = await computeCyclicLoss(model, vocab, invVocab, ipa, word, ">");
  const tableData = cyclicData.details.map((row, index) => ({
    Step: index + 1,
    Char: row.char,
    Prob: row.prob,
    Loss: row.loss,
  }));

  console.log(
    `%cCyclic Loss Analysis for "${word}"`,
    "font-weight: bold; font-size: 14px;",
  );
  console.table(tableData);

  document.getElementById("output").innerHTML = `
<strong>IPA:</strong> ${ipa}
<br>
<strong>Reversed Word:</strong> ${back_word}
<br>
<strong>Avg Cyclic Loss:</strong> ${cyclicData.avgLoss}
<br>
(Lower is better. High loss on specific chars indicates phonetic ambiguity.)
`;
}

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