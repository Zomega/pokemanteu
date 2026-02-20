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
