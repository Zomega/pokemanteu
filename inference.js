// --- CONFIGURATION ---
// In the browser, these are URLs relative to the HTML file
const MODEL_URL = "./tfjs_model/model.json";
const VOCAB_URL = "./vocab.json";
const MAX_LEN = 40;
const BEAM_WIDTH = 5;

let model = null;
let vocab = null;
let invVocab = null;

// --- SETUP ---
async function loadResources() {
  if (model) return; // Already loaded

  console.log("Loading model...");
  // BROWSER DIFFERENCE 1: automatic fetching
  // tf.loadGraphModel automatically uses 'fetch' in the browser.
  // No fileLoader or custom handlers needed!
  model = await tf.loadGraphModel(MODEL_URL);

  console.log("Loading vocab...");
  const vocabResp = await fetch(VOCAB_URL);
  vocab = await vocabResp.json();
  invVocab = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));

  console.log("Ready!");
}

async function runInference() {
  await loadResources();
  const word = document.getElementById("inputWord").value.trim();

  // 1. Get the IPA via your existing Beam Search
  const ipa = await decodeBeamBatched(word, "<");
  const back_word = await decodeBeamBatched(ipa, ">");

  // 2. Calculate the "Force-Fed" loss of the word given that IPA
  const cyclicData = await computeCyclicLoss(ipa, word, ">");

  // 3. Render Table
  // We map it just to capitalize keys or add an index for a cleaner table
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
                            <small>(Lower is better. High loss on specific chars indicates phonetic ambiguity.)</small>
          `;
}

async function decodeBeamBatched(word, taskToken, alpha = 0.6) {
  // PHASE 1: Generate Candidates (Forward Pass)
  const candidates = tf.tidy(() => {
    const fullText = taskToken + word.toLowerCase();
    const START_TOKEN = vocab["["];
    const STOP_TOKEN = vocab["]"];
    const PAD_TOKEN = vocab["[PAD]"];

    const getAlphaScore = (logProb, length) => {
      const lp = Math.pow((5 + length) / 6, alpha);
      return logProb / lp;
    };

    let encIds = fullText
      .split("")
      .map((c) => vocab[c] || 0)
      .slice(0, MAX_LEN);
    while (encIds.length < MAX_LEN) encIds.push(PAD_TOKEN);

    const encTensor = tf.tensor2d(
      Array(BEAM_WIDTH).fill(encIds),
      [BEAM_WIDTH, MAX_LEN],
      "float32",
    );
    let scores = tf.tensor1d([0.0, ...Array(BEAM_WIDTH - 1).fill(-1e9)]);
    let sequences = tf.fill([BEAM_WIDTH, 1], START_TOKEN, "int32");

    let finishedSeqs = [];
    let finishedScores = [];

    for (let i = 0; i < MAX_LEN - 1; i++) {
      const currLen = sequences.shape[1];
      const padSize = MAX_LEN - 1 - currLen;
      let decInput = sequences.cast("float32");
      if (padSize > 0)
        decInput = decInput.pad(
          [
            [0, 0],
            [0, padSize],
          ],
          PAD_TOKEN,
        );

      const inputs = { enc_in: encTensor, dec_in: decInput };
      let preds = model.execute(inputs);
      if (Array.isArray(preds)) preds = preds[0];

      const nextTokenLogits = preds
        .gather([currLen - 1], 1)
        .reshape([BEAM_WIDTH, -1]);
      const logProbs = tf.log(nextTokenLogits.add(1e-9));

      const candidateScores = scores.expandDims(1).add(logProbs);
      const flatScores = candidateScores.reshape([-1]);
      const { values: topKScores, indices: topKIndices } = tf.topk(
        flatScores,
        BEAM_WIDTH,
      );

      const vocabSize = logProbs.shape[1];
      const beamIndices = topKIndices
        .div(tf.scalar(vocabSize, "int32"))
        .cast("int32");
      const tokenIndices = topKIndices
        .mod(tf.scalar(vocabSize, "int32"))
        .cast("int32");

      const bIdxArr = beamIndices.arraySync();
      const tIdxArr = tokenIndices.arraySync();
      const sArr = topKScores.arraySync();
      const seqsArr = sequences.arraySync();

      const nextSeqs = [];
      const nextScoresArr = [];

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

      sequences = tf.tensor2d(
        nextSeqs,
        [BEAM_WIDTH, nextSeqs[0].length],
        "int32",
      );
      scores = tf.tensor1d(nextScoresArr);

      if (finishedSeqs.length >= BEAM_WIDTH) {
        const finishedWithAlpha = finishedScores.map((s, idx) => {
          const len = finishedSeqs[idx].length - 1;
          return getAlphaScore(s, len);
        });
        finishedWithAlpha.sort((a, b) => b - a);
        const worstWinningScore = finishedWithAlpha[BEAM_WIDTH - 1];
        const bestActiveLogProb = Math.max(...nextScoresArr);

        if (bestActiveLogProb > -1e8) {
          const maxPossibleLP = Math.pow((5 + MAX_LEN) / 6, alpha);
          if (bestActiveLogProb / maxPossibleLP < worstWinningScore) break;
        } else {
          break;
        }
      }
    }

    const finalActiveSeqs = sequences.arraySync();
    const finalActiveScores = scores.arraySync();
    let allCandidates = [];
    for (let i = 0; i < finishedSeqs.length; i++) {
      allCandidates.push({
        seq: finishedSeqs[i],
        rawScore: finishedScores[i],
        status: "DONE",
      });
    }
    for (let i = 0; i < finalActiveSeqs.length; i++) {
      if (finalActiveScores[i] > -1e8) {
        allCandidates.push({
          seq: finalActiveSeqs[i],
          rawScore: finalActiveScores[i],
          status: "MAX_LEN",
        });
      }
    }

    return allCandidates
      .map((c) => {
        const cleanIds = c.seq.filter(
          (id) => id !== START_TOKEN && id !== STOP_TOKEN && id !== PAD_TOKEN,
        );
        const text = cleanIds.map((id) => invVocab[id]).join("");
        const alphaScore = getAlphaScore(c.rawScore, cleanIds.length);

        return {
          Text: text,
          "Alpha Score": alphaScore,
          "Raw LogProb": c.rawScore,
          Status: c.status,
        };
      })
      .sort((a, b) => b["Alpha Score"] - a["Alpha Score"]);
  });

  // PHASE 2: Check Cyclic Loss (Backward Pass)

  // 1. Determine the opposite direction
  // If input was < (G2P), backward is > (P2G)
  // If input was > (P2G), backward is < (G2P)
  const backwardTaskToken = taskToken === "<" ? ">" : "<";

  // 2. Compute Loss
  const detailedCandidates = await Promise.all(
    candidates.map(async (c) => {
      // We feed the Generated Text (c.Text) + backward token to see if it generates the Original (word)
      const lossData = await computeCyclicLoss(c.Text, word, backwardTaskToken);

      return {
        ...c,
        "Cyclic Loss": lossData.avgLoss,
      };
    }),
  );

  // 3. Log
  const tableData = detailedCandidates.map((row, idx) => ({
    Rank: idx + 1,
    Text: row.Text,
    "Alpha Score": row["Alpha Score"].toFixed(4),
    "Cyclic Loss": row["Cyclic Loss"],
    Status: row.Status,
  }));

  console.log(
    `%cBeam Search Analysis (${taskToken} -> ${backwardTaskToken})`,
    "font-weight: bold; color: #4CAF50",
  );
  console.table(tableData);

  return detailedCandidates.length > 0 ? detailedCandidates[0].Text : "";
}

// ADD backwardTaskToken as the 3rd argument
async function computeCyclicLoss(
  generatedOutput,
  originalInput,
  backwardTaskToken,
) {
  return tf.tidy(() => {
    const PAD_TOKEN = vocab["[PAD]"];

    // 1. Prepare Encoder Input (The output from the forward pass)
    // We use the backwardTaskToken here (e.g., if checking a word, we use '<' to see if it generates IPA)
    const encText = backwardTaskToken + generatedOutput.toLowerCase();
    let encIds = encText
      .split("")
      .map((c) => vocab[c] || 0)
      .slice(0, MAX_LEN);
    while (encIds.length < MAX_LEN) encIds.push(PAD_TOKEN);
    const encTensor = tf.tensor2d([encIds], [1, MAX_LEN]);

    // 2. Prepare Decoder Input (The original ground truth input)
    const fullTargetText = "[" + originalInput.toLowerCase() + "]";
    let allTargetIds = fullTargetText.split("").map((c) => vocab[c] || 0);

    let decInputIds = allTargetIds.slice(0, MAX_LEN - 1);
    while (decInputIds.length < MAX_LEN - 1) decInputIds.push(PAD_TOKEN);

    const decInput = tf.tensor2d([decInputIds], [1, MAX_LEN - 1], "float32");

    // 3. Run Model
    let preds = model.execute({ enc_in: encTensor, dec_in: decInput });
    if (Array.isArray(preds)) preds = preds[0];

    let totalLoss = 0;
    let charResults = [];
    const limit = Math.min(allTargetIds.length - 1, MAX_LEN - 1);

    for (let i = 0; i < limit; i++) {
      const nextCharId = allTargetIds[i + 1];
      const stepProbs = preds.gather([i], 1).reshape([-1]);
      const charProb = stepProbs.gather([nextCharId]).dataSync()[0];

      const safeProb = Math.max(charProb, 1e-9);
      const stepLoss = -Math.log(safeProb);

      totalLoss += stepLoss;
      charResults.push({
        char: invVocab[nextCharId],
        prob: (charProb * 100).toFixed(2) + "%",
        loss: stepLoss.toFixed(4),
      });
    }

    return {
      avgLoss: (totalLoss / limit).toFixed(4),
      details: charResults,
    };
  });
}

// Auto-load on page open
loadResources();
