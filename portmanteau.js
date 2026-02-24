/**
 * Portmanteau linguistic engine for Pokémon name generation.
 */

// We'll populate these from your creatures.json during initialization
let VALID_NONVOWEL_CLUSTERS = new Set();
let VALID_VOWEL_CLUSTERS = new Set();
// TODO: We should do this by loading a markov model, rather than hacking it in initLinguisticEngine.
let BIGRAM_FREQUENCY = {};
let BIGRAM_THRESHOLD = 1;

// Configuration Constants
const TOO_LITTLE_PENALTY = 10;
const TOO_MUCH_PENALTY = 1;
const SHORT_LEN_PENALTY = [5, 2, 1, 0.6, 0.3, 0.3];

/** * Basic Syllable counter (JS version of hacky_syllables)
 */
function countSyllables(word) {
    word = word.toLowerCase();
    if (word.length <= 3) return 1;
    word = word.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, '');
    word = word.replace(/^y/, '');
    const res = word.match(/[aeiouy]{1,2}/g);
    return res ? res.length : 1;
}

/**
 * Scoring Criteria
 */
const RankingCriteria = {
    charLength: (item, min, max) => {
        const diff = Math.max(0, min - item.length, item.length - max);
        return diff <= 0 ? 0 : -(Math.pow(2, diff));
    },

    clusters: (item, regex, validSet) => {
        const matches = item.toLowerCase().match(regex) || [];
        const invalid = matches.filter(c => !validSet.has(c));
        return -invalid.length;
    },

    bigrams: (item) => {
        let penalty = 0;
        let weirdCount = 0;
        for (let i = 0; i < item.length - 1; i++) {
            const bigram = item.substring(i, i + 2).toLowerCase();
            const freq = (BIGRAM_FREQUENCY[bigram] || 0) / BIGRAM_THRESHOLD;
            if (freq < 1) {
                weirdCount++;
                penalty += Math.pow(1 - freq, 2);
            }
        }
        return weirdCount === 0 ? 0 : -penalty / weirdCount;
    },

    commonality: (item, target) => {
        // Simplified LCS logic for performance
        let lcs = 0;
        for (let i = 0; i < item.length; i++) {
            if (target.includes(item[i])) lcs++;
        }
        let penalty = 0;
        if (lcs * 4 < target.length) penalty += TOO_LITTLE_PENALTY;
        if (lcs * 4 < target.length * 3) penalty += TOO_MUCH_PENALTY;
        return -penalty;
    }
};

/**
 * Main Candidate Generator
 * TODO: This can produce duplicates! We should dedupe before ranking.
 */
export function generatePortmanteaus(s1, s2, topN = 50) {
    const candidates = [];

    const minSize = Math.min(s1.length, s2.length);
    const maxSize = Math.max(s1.length, s2.length);

    for (let l1 = 1; l1 < s1.length; l1++) {
        for (let l2 = 1; l2 < s2.length; l2++) {
            const word = (s1.substring(0, l1 + 1) + s2.substring(s2.length - l2 - 1)).toLowerCase();

            // Calculate Composite Score
            let score = 0;
            score += 0.3 * RankingCriteria.charLength(word, minSize, maxSize);
            score += 7.0 * RankingCriteria.clusters(word, /[^aeiouy]{2,}/g, VALID_NONVOWEL_CLUSTERS);
            score += 4.0 * RankingCriteria.clusters(word, /[aeiouy]{2,}/g, VALID_VOWEL_CLUSTERS);
            score += 4.0 * RankingCriteria.bigrams(word);
            score += 0.6 * RankingCriteria.commonality(word, s1);
            score += 0.6 * RankingCriteria.commonality(word, s2);
            score += 0.5 * (-countSyllables(word));

            candidates.push({
                name: word.charAt(0).toUpperCase() + word.slice(1),
                score
            });
        }
    }

    // Sort by score descending and return top N
    return candidates.sort((a, b) => b.score - a.score).slice(0, topN);
}

/**
 * Initialization: Feeds valid clusters from your existing creature database
 */
export function initLinguisticEngine(rootNode) {
    const allWords = [];

    function walk(node) {
        if (node.word && !node.unsearchable) allWords.push(node.word.toLowerCase());
        if (node.children) node.children.forEach(walk);
    }
    walk(rootNode);

    allWords.forEach(word => {
        // Extract Bigrams
        for (let i = 0; i < word.length - 1; i++) {
            const b = word.substring(i, i + 2);
            BIGRAM_FREQUENCY[b] = (BIGRAM_FREQUENCY[b] || 0) + 1;
        }
        // Extract Clusters
        (word.match(/[^aeiouy]{2,}/g) || []).forEach(c => VALID_NONVOWEL_CLUSTERS.add(c));
        (word.match(/[aeiouy]{2,}/g) || []).forEach(c => VALID_VOWEL_CLUSTERS.add(c));
    });

    const totalBigrams = Object.values(BIGRAM_FREQUENCY).reduce((a, b) => a + b, 0);
    BIGRAM_THRESHOLD = totalBigrams * 0.003;
}