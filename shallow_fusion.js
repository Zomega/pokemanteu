// --- 1. LOGITS PROCESSORS (Inside the Loop) ---
// These modify the probability distribution of the *next* token.
export class LogitsProcessor {
  constructor(name) {
    this.name = name;
  }
  
  /**
   * @param {number[][]} sequencesArr - The current token IDs for each beam.
   * @param {number} vocabSize - The size of the vocabulary.
   * @param {Object} context - Contextual data (vocab maps, task tokens, etc.)
   * @returns {tf.Tensor2D | null} - A [beam_width, vocab_size] tensor of log-prob adjustments, or null.
   */
  process(sequencesArr, vocabSize, context) {
    return null; 
  }
}

// --- 2. SEQUENCE SCORERS (Outside the Loop) ---
// These evaluate a fully generated string and return a final score penalty/bonus.
export class SequenceScorer {
  constructor(name) {
    this.name = name;
  }

  /**
   * @param {Object[]} candidates - Array of finished sequence objects {Text, "Alpha Score", etc.}
   * @param {Object} context - Contextual data.
   * @returns {Promise<number[] | null>} - Array of score modifiers aligning with the candidates.
   */
  async score(candidates, context) {
    return null;
  }
}