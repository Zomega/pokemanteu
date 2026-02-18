const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

// --- CONFIGURATION ---
const MODEL_DIR = './tfjs_model';
const MODEL_FILE = 'model.json';
const VOCAB_FILE = './vocab.json';
const MAX_LEN = 40;
const BEAM_WIDTH = 5;

// --- SIMPLE LOADER ---
const fileLoader = {
  load: async () => {
    const modelPath = path.join(MODEL_DIR, MODEL_FILE);
    const modelJson = JSON.parse(fs.readFileSync(modelPath, 'utf8'));

    const weightFileName = modelJson.weightsManifest[0].paths[0];
    const weightsPath = path.join(MODEL_DIR, weightFileName);
    const weightsBuffer = fs.readFileSync(weightsPath);

    return {
      modelTopology: modelJson.modelTopology || modelJson,
      weightSpecs: modelJson.weightsManifest[0].weights,
      weightData: weightsBuffer.buffer
    };
  }
};

async function runInference() {
  try {
    console.log("Loading model...");

    // UPDATED: Use loadGraphModel for the new format
    const model = await tf.loadGraphModel(fileLoader);

    // Load Vocab
    const vocab = JSON.parse(fs.readFileSync(VOCAB_FILE, 'utf8'));
    const invVocab = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));

    console.log("--------------------------------");
    console.log("Model Loaded. Running Inference.");
    console.log("--------------------------------");

    // Test Case
    const inputWord = "Pikachu";
    const ipa = await decodeBeamBatched(model, inputWord, "<", vocab, invVocab);

    console.log(`INPUT:  ${inputWord}`);
    console.log(`OUTPUT: ${ipa}`);

  } catch (err) {
    console.error("Error:", err.message);
  }
}

// --- BEAM SEARCH LOGIC ---
async function decodeBeamBatched(model, word, taskToken, vocab, invVocab) {
    return tf.tidy(() => {
      const fullText = taskToken + word.toLowerCase();
      const START_TOKEN = vocab["["];
      const STOP_TOKEN = vocab["]"];
      const PAD_TOKEN = vocab["[PAD]"];

      let encIds = fullText.split('').map(c => vocab[c] || 0).slice(0, MAX_LEN);
      while (encIds.length < MAX_LEN) encIds.push(PAD_TOKEN);

      const encTensor = tf.tensor2d(Array(BEAM_WIDTH).fill(encIds), [BEAM_WIDTH, MAX_LEN], 'float32');
      let scores = tf.tensor1d([0.0, ...Array(BEAM_WIDTH - 1).fill(-1e9)]);
      let sequences = tf.fill([BEAM_WIDTH, 1], START_TOKEN, 'int32');

      let finishedSeqs = [];
      let finishedScores = [];

      for (let i = 0; i < MAX_LEN - 1; i++) {
        const currLen = sequences.shape[1];
        const padSize = (MAX_LEN - 1) - currLen;

        let decInput = sequences;
        // If we need to cast sequences (which tracks IDs as ints) to float for the model:
        let modelInput = decInput.cast('float32');

        if (padSize > 0) {
            // Pad the float tensor
            modelInput = modelInput.pad([[0, 0], [0, padSize]], PAD_TOKEN);
        }

        // Execute with the float inputs
        const inputs = {
            'enc_in': encTensor,
            'dec_in': modelInput
        };

        let preds = model.execute(inputs);

        if (Array.isArray(preds)) preds = preds[0];

        const nextTokenLogits = preds.gather([currLen - 1], 1).reshape([BEAM_WIDTH, -1]);
        const logProbs = tf.logSoftmax(nextTokenLogits);

        const candidateScores = scores.expandDims(1).add(logProbs);
        const flatScores = candidateScores.reshape([-1]);
        const {values: topKScores, indices: topKIndices} = tf.topk(flatScores, BEAM_WIDTH);

        const vocabSize = logProbs.shape[1];
        const beamIndices = topKIndices.div(tf.scalar(vocabSize, 'int32')).cast('int32');
        const tokenIndices = topKIndices.mod(tf.scalar(vocabSize, 'int32')).cast('int32');

        const nextSeqs = [];
        const nextScoresArr = [];

        // Sync to CPU for list manipulation
        const bIdxArr = beamIndices.arraySync();
        const tIdxArr = tokenIndices.arraySync();
        const sArr = topKScores.arraySync();
        const seqsArr = sequences.arraySync();

        for (let k = 0; k < BEAM_WIDTH; k++) {
          const bIdx = bIdxArr[k];
          const token = tIdxArr[k];
          const score = sArr[k];

          if (token === STOP_TOKEN) {
            finishedSeqs.push(seqsArr[bIdx]);
            finishedScores.push(score);
            nextSeqs.push([...seqsArr[bIdx], PAD_TOKEN]);
            nextScoresArr.push(-1e9);
          } else {
            nextSeqs.push([...seqsArr[bIdx], token]);
            nextScoresArr.push(score);
          }
        }

        sequences = tf.tensor2d(nextSeqs, [BEAM_WIDTH, nextSeqs[0].length], 'int32');
        scores = tf.tensor1d(nextScoresArr);

        if (tf.max(scores).arraySync() < -1e8) break;
      }

      let finalSeq;
      if (finishedSeqs.length > 0) {
        const bestIdx = finishedScores.indexOf(Math.max(...finishedScores));
        finalSeq = finishedSeqs[bestIdx];
      } else {
        const bestIdx = scores.argMax().dataSync()[0];
        finalSeq = sequences.arraySync()[bestIdx];
      }

      return finalSeq
        .filter(id => id !== START_TOKEN && id !== STOP_TOKEN && id !== PAD_TOKEN)
        .map(id => invVocab[id])
        .join('');
    });
}

runInference();