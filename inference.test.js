import { describe, it, expect, vi, beforeAll } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { InferenceEngine } from './inference.js';

const mockVocab = {
  "[": 0, "]": 1, "[PAD]": 2, "<": 3,
  "p": 4, "i": 5, "k": 6, "a": 7, "c": 8, "h": 9, "u": 10,
  "ɪ": 11, "ə": 12, "t": 13, "ʃ": 14
};

/**
 * Creates a mock prediction tensor that steers the beam search
 * toward a specific sequence of tokens.
 */
const createSmartPrediction = (stepIndex, vocabSize) => {
  // Sequence we want to force: p (4), ɪ (11), k (6), ə (12), ] (1)
  const targetSequence = [4, 11, 6, 12, 1];

  return tf.tidy(() => {
    // Create a base of very low probability (using -10 for log-space feel)
    const logits = tf.fill([1, 40, vocabSize], -10.0);

    // For each beam search step, give the target token a massive boost
    const buffer = logits.bufferSync();
    if (stepIndex < targetSequence.length) {
      const targetToken = targetSequence[stepIndex];
      // Set the probability of our target token to ~1.0 (0 in log space)
      // We apply this to all time steps in the output to ensure gather() hits it
      for (let t = 0; t < 40; t++) {
        buffer.set(0, 0, t, targetToken);
      }
    }

    // Convert back to "probabilities" (softmax) so the engine's tf.log() works
    return tf.softmax(logits);
  });
};

describe('InferenceEngine Functional Smoke Test', () => {
  beforeAll(async () => {
    await tf.ready();
  });

  it('should correctly decode the forced IPA sequence "pɪkə"', async () => {
    let callCount = 0;

    const smartModel = {
      execute: vi.fn(() => {
        const tensor = createSmartPrediction(callCount, Object.keys(mockVocab).length);
        callCount++;
        return tensor;
      })
    };

    const engine = new InferenceEngine(smartModel, mockVocab, 40);

    // Act: Run with beam_width 1 to trace the forced path
    const result = await engine.decodeBeamBatched("Pikachu", "<", 1);

    // Assert
    expect(result).toBe("pɪkə");
    console.log('Decoded Result:', result);
  });
});