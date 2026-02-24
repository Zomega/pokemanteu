// TODO: It's possible this can be integrated with inference.js, but I'd also like to use it in non ML contexts, e.g. to steer a markov model or in portmanteau.js

// --- 1. LOGITS PROCESSORS (Inside the Loop) ---
// These modify the probability distribution of the *next* token.
// TODO: Enable "constraint" processors, which can fully reject a beam with certian properties?
// This can be done with a very negative score, but a hard respected options would be good.
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
// TODO: Enable constraints which hard reject options.
// This can be done with a very negative score, but a hard respected options would be good.
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