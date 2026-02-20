import * as tf from 'https://esm.run/@tensorflow/tfjs';
import {SequenceScorer} from './shallow_fusion.js';

// TODO: Include this in the model context?
const MAX_LEN = 40;

export async function decodeBeamBatched(
  model,
  vocab,
  invVocab,
  word,
  taskToken,
  beam_width = 3,
  temperature = 1.0,
  alpha = 0.6,
  logitsProcessors = [], // Array of { processor: LogitsProcessor, weight: number }
  sequenceScorers = []   // Array of { scorer: SequenceScorer, weight: number }
) {
  const fusionContext = { originalInput: word, taskToken, vocab, invVocab };
  // TODO: Plumn in logitsProcessors and sequenceScorers

  // TODO: Remove once no longer used.
  const lambda_bidir_rerank = 6;
  // PHASE 1: Generate Candidates (Forward Pass)
  const candidates = tf.tidy(() => {
    let cleanInput = word;
    if (taskToken === "<") {
      cleanInput = word.toLowerCase();
    }
    const fullText = taskToken + cleanInput;
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
      Array(beam_width).fill(encIds),
      [beam_width, MAX_LEN],
      "float32",
    );
    let scores = tf.tensor1d([0.0, ...Array(beam_width - 1).fill(-1e9)]);
    let sequences = tf.fill([beam_width, 1], START_TOKEN, "int32");

    let finishedSeqs = [];
    let finishedScores = [];

    // TODO: Ensure that we can track which logitsProcessors are scoring what after the fact.
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
        .reshape([beam_width, -1]);
      const logProbs = tf.log(nextTokenLogits.add(1e-9));

      // >>> 2. APPLY TEMPERATURE SCALING >>>
      // TODO: Seems like we should apply regardless?
      if (temperature !== 1.0) {
        const scaledLogProbs = tf.tidy(() => {
          // Divide the log-probabilities by the temperature
          const scaled = logProbs.div(temperature);
          // Re-normalize using logSoftmax so they remain valid probabilities
          return tf.logSoftmax(scaled);
        });

        logProbs.dispose(); // Prevent memory leak
        logProbs = scaledLogProbs; // Reassign
      }
      // <<< END TEMPERATURE <<<

      const candidateScores = scores.expandDims(1).add(logProbs);
      const flatScores = candidateScores.reshape([-1]);
      const { values: topKScores, indices: topKIndices } = tf.topk(
        flatScores,
        beam_width,
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

      for (let k = 0; k < beam_width; k++) {
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
        [beam_width, nextSeqs[0].length],
        "int32",
      );
      scores = tf.tensor1d(nextScoresArr);

      if (finishedSeqs.length >= beam_width) {
        const finishedWithAlpha = finishedScores.map((s, idx) => {
          const len = finishedSeqs[idx].length - 1;
          return getAlphaScore(s, len);
        });
        finishedWithAlpha.sort((a, b) => b - a);
        const worstWinningScore = finishedWithAlpha[beam_width - 1];
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

  // --- PHASE 2 & 3: Sequence Scorers & Reranking ---

  // 1. Initialize Fused Score with the Forward Alpha Score
  candidates.forEach(c => {
    c["Fused Score"] = parseFloat(c["Alpha Score"]);
  });

  // 2. Apply each Sequence Scorer dynamically
  // Note: fusionContext should be defined at the top of decodeBeamBatched as:
  // const fusionContext = { originalInput: word, taskToken, vocab, invVocab };
  for (const { scorer, weight } of sequenceScorers) {
    if (weight === 0) continue;

    // Get an array of modifiers (e.g., negative cyclic loss) parallel to the candidates
    const finishScores = await scorer.score(candidates, fusionContext);

    if (finishScores) {
      for (let i = 0; i < candidates.length; i++) {
        const scoreDelta = weight * finishScores[i];

        // Add the weighted modifier to the fused score
        candidates[i]["Fused Score"] += scoreDelta;

        // Save the raw score on the candidate for logging purposes
        candidates[i][`ScorerRaw_${scorer.name}`] = finishScores[i].toFixed(4);
      }
    }
  }

  // 3. Re-sort candidates based on the new Fused Score (Descending)
  candidates.sort((a, b) => b["Fused Score"] - a["Fused Score"]);

  // 4. Build and log the detailed reranking table dynamically
  const tableData = candidates.map((row, idx) => {
    // Base columns that are always present
    const tableRow = {
      "Final Rank": idx + 1,
      "Text": row.Text,
      "Fused Score": row["Fused Score"].toFixed(4),
      "Fwd Alpha": row["Alpha Score"].toFixed(4),
      "Status": row.Status,
    };

    // Dynamically append columns for any active scorers
    for (const { scorer, weight } of sequenceScorers) {
      if (weight !== 0 && row[`ScorerRaw_${scorer.name}`] !== undefined) {
        // e.g. Creates a column named "CyclicLoss (w=10)"
        tableRow[`${scorer.name} (w=${weight})`] = row[`ScorerRaw_${scorer.name}`];
      }
    }

    return tableRow;
  });

  // Create a dynamic title string showing which scorers were used
  const activeScorersStr = sequenceScorers
    .filter(s => s.weight !== 0)
    .map(s => `${s.scorer.name}=${s.weight}`)
    .join(" | ");

  console.log(
    `%cPost-Hoc Reranking (${taskToken}) | Scorers: ${activeScorersStr || "None"}`,
    "font-weight: bold; color: #FF9800; font-size: 12px;",
  );
  console.table(tableData);

  // --- RETURN BEST ---
  // We now return the top-ranked item after fusion.
  return candidates.length > 0 ? candidates[0].Text : "";
}

export async function computeCyclicLossBatch(
  model,
  vocab,
  invVocab,
  generatedCandidates,
  originalInput,
  backwardTaskToken,
) {
    // TODO: Can we use the length of the originalInput here rather than MAX_LEN?
  return tf.tidy(() => {
    const BATCH_SIZE = generatedCandidates.length;
    const PAD_TOKEN = vocab["[PAD]"];

    // --- 1. Prepare Encoder Inputs ---
    const encIdsBatch = generatedCandidates.map((txt) => {
      let cleanTxt = txt;
      if (backwardTaskToken === "<") {
        cleanTxt = txt.toLowerCase();
      }
      const encText = backwardTaskToken + cleanTxt;
      let ids = encText
        .split("")
        .map((c) => vocab[c] || 0)
        .slice(0, MAX_LEN);
      while (ids.length < MAX_LEN) ids.push(PAD_TOKEN);
      return ids;
    });
    const encTensor = tf.tensor2d(
      encIdsBatch,
      [BATCH_SIZE, MAX_LEN],
      "float32",
    );

    // --- 2. Prepare Decoder Inputs ---
    let targetTxt = originalInput;
    if (backwardTaskToken === ">") {
      targetTxt = originalInput.toLowerCase();
    }

    const fullTargetText = "[" + targetTxt + "]";
    let allTargetIds = fullTargetText.split("").map((c) => vocab[c] || 0);

    let decInputIds = allTargetIds.slice(0, MAX_LEN - 1);
    while (decInputIds.length < MAX_LEN - 1) decInputIds.push(PAD_TOKEN);

    const decInputSingle = tf.tensor2d(
      [decInputIds],
      [1, MAX_LEN - 1],
      "float32",
    );
    const decInputBatch = decInputSingle.tile([BATCH_SIZE, 1]);

    // --- 3. Run Model ---
    let preds = model.execute({ enc_in: encTensor, dec_in: decInputBatch });
    if (Array.isArray(preds)) preds = preds[0];

    // --- 4. Compute Loss ---
    const results = [];
    const limit = Math.min(allTargetIds.length - 1, MAX_LEN - 1);

    for (let b = 0; b < BATCH_SIZE; b++) {
      // Get prediction matrix for this batch item: [SEQ_LEN, VOCAB_SIZE]
      const batchPreds = preds.gather([b], 0).squeeze([0]);

      let totalLoss = 0;
      const charDetails = [];

      for (let i = 0; i < limit; i++) {
        const nextCharId = allTargetIds[i + 1];

        const charProb = batchPreds
          .slice([i, nextCharId], [1, 1])
          .dataSync()[0];

        const safeProb = Math.max(charProb, 1e-9);
        const stepLoss = -Math.log(safeProb);

        totalLoss += stepLoss;

        charDetails.push({
          char: invVocab[nextCharId],
          prob: (charProb * 100).toFixed(2) + "%",
          loss: stepLoss.toFixed(4),
        });
      }
      results.push({
        avgLoss: (totalLoss / limit).toFixed(4),
        details: charDetails,
      });
    }
    return results;
  });
}

export async function computeCyclicLoss(
  model,
  vocab,
  invVocab,
  generatedIpa,
  originalWord,
  backwardTaskToken,
) {
  const results = await computeCyclicLossBatch(
    model,
    vocab,
    invVocab,
    [generatedIpa],
    originalWord,
    backwardTaskToken,
  );
  return results[0];
}


// The Cyclic Loss model ONLY scores finished sequences.
// TODO: Link up properly.
export class CyclicSequenceScorer extends SequenceScorer {
  constructor(model, vocab, invVocab, computeCyclicLossBatchFn) {
    super("CyclicLoss");
    this.model = model;
    this.vocab = vocab;
    this.invVocab = invVocab;
    this.computeLoss = computeCyclicLossBatchFn;
  }

  async score(candidates, context) {
    const backwardTaskToken = context.taskToken === "<" ? ">" : "<";
    const textInputs = candidates.map(c => c.Text);

    const lossResults = await this.computeLoss(this.model, this.vocab, this.invVocab, textInputs, context.originalInput, backwardTaskToken);
    return lossResults.map(r => -parseFloat(r.avgLoss));
  }
}