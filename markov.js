import * as tf from 'https://esm.run/@tensorflow/tfjs';
import {LogitsProcessor} from './shallow_fusion.js';

export class MultiMarkovGenerator {
  constructor(jsonData) {
    this.order = jsonData.order;
    this.models = {};
    this.START_TOKEN = "<START>";
    this.END_TOKEN = "<END>";

    // CRITICAL FIX: Python and JS space JSON arrays differently.
    // Python: '["<START>", "p"]' | JS: '["<START>","p"]'
    // We parse and re-stringify the keys on load to ensure JS compatibility.
    for (const [modelName, states] of Object.entries(jsonData.models)) {
      this.models[modelName] = {};
      for (const [stateStr, counts] of Object.entries(states)) {
        const parsedState = JSON.parse(stateStr);
        const normalizedKey = JSON.stringify(parsedState);
        this.models[modelName][normalizedKey] = counts;
      }
    }
    console.log("Markov Models loaded:", Object.keys(this.models));
  }

  _getFusedProbabilities(state, weights) {
    const stateKey = JSON.stringify(state);
    let activeModels = [];
    let totalActiveWeight = 0;

    // 1. Find which models actually know this state
    for (const [name, weight] of Object.entries(weights)) {
      if (this.models[name] && this.models[name][stateKey] && weight > 0) {
        const counts = this.models[name][stateKey];
        const total = Object.values(counts).reduce((a, b) => a + b, 0);
        if (total > 0) {
          activeModels.push({ name, counts, total, weight });
          totalActiveWeight += weight;
        }
      }
    }

    // Dead end: No models know this state
    if (activeModels.length === 0) return null;

    // 2. Gather unique characters across all active models
    const allChars = new Set();
    activeModels.forEach((m) =>
      Object.keys(m.counts).forEach((char) => allChars.add(char)),
    );

    const population = [];
    const probabilities = [];

    // 3. Calculate fused probabilities with dynamic fallback
    for (const char of allChars) {
      let fusedP = 0.0;
      for (const m of activeModels) {
        const normalizedW = m.weight / totalActiveWeight;
        const count = m.counts[char] || 0;
        const pChar = count / m.total;
        fusedP += normalizedW * pChar;
      }
      population.push(char);
      probabilities.push(fusedP);
    }

    return { population, probabilities };
  }

  _weightedChoice(items, weights) {
    let rand = Math.random();
    for (let i = 0; i < items.length; i++) {
      if (rand < weights[i]) return items[i];
      rand -= weights[i];
    }
    return items[items.length - 1]; // Safe fallback
  }

  generate(weights, minLength = 4, maxLength = 12) {
    let currentState = Array(this.order).fill(this.START_TOKEN);
    let generatedChars = [];

    while (generatedChars.length < maxLength) {
      const probs = this._getFusedProbabilities(currentState, weights);

      if (!probs) break; // Dead end hit

      const nextChar = this._weightedChoice(
        probs.population,
        probs.probabilities,
      );

      if (nextChar === this.END_TOKEN) {
        if (generatedChars.length >= minLength) {
          break;
        } else {
          // Too short, restart recursively
          return this.generate(weights, minLength, maxLength);
        }
      }

      generatedChars.push(nextChar);
      currentState.shift(); // Remove oldest token
      currentState.push(nextChar); // Add newest token
    }

    return generatedChars.join("");
  }
}


export class MarkovLogitsProcessor extends LogitsProcessor {
  constructor(markovGenerator, markovWeights) {
    super("MarkovFusion");
    this.generator = markovGenerator;
    this.weights = markovWeights;
  }

  process(sequencesArr, vocabSize, context) {
    // Only apply during IPA generation
    if (context.taskToken !== "<") return null;

    const beam_width = sequencesArr.length;
    const markovBatch = [];

    // Create a harsh penalty for completely impossible transitions
    // (e.g., -20 is a massive log-prob penalty)
    const BASE_PENALTY = -20.0;

    for (let b = 0; b < beam_width; b++) {
      const beamSeq = sequencesArr[b];
      let stateChars = [];

      for (let j = Math.max(0, beamSeq.length - this.generator.order); j < beamSeq.length; j++) {
        let char = context.invVocab[beamSeq[j]];
        if (char === "[") char = this.generator.START_TOKEN;
        stateChars.push(char);
      }
      while (stateChars.length < this.generator.order) {
        stateChars.unshift(this.generator.START_TOKEN);
      }

      const probsInfo = this.generator._getFusedProbabilities(stateChars, this.weights);

      // Initialize row with the base penalty
      let markovArr = new Array(vocabSize).fill(BASE_PENALTY);

      if (probsInfo && probsInfo.population.length > 0) {
        for (let v = 0; v < probsInfo.population.length; v++) {
          const char = probsInfo.population[v];
          const vocabIdx = context.vocab[char];
          if (vocabIdx !== undefined) {
             // Convert probability [0, 1] to log-prob (-inf, 0].
             // Add small epsilon so Math.log(0) doesn't return -Infinity.
            markovArr[vocabIdx] = Math.log(probsInfo.probabilities[v] + 1e-9);
          }
        }
      } else {
         // DEAD END: If the Markov model has NO IDEA what comes next,
         // don't penalize anything. Let the Keras model decide.
         markovArr.fill(0.0);
      }
      markovBatch.push(markovArr);
    }
    return tf.tensor2d(markovBatch, [beam_width, vocabSize], "float32");
  }
}