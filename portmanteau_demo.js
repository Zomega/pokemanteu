import {
    FuzzyMatcher as fuzz, // TODO: Unused? We should use it...
    BitapScanner,
    SymSpellIndex
} from './fuzzy_string.js';

import {
    initLinguisticEngine,
    generatePortmanteaus
} from './portmanteau.js';

let uniquenessIndex;

/**
 * Primes the SymSpell index with all existing names to prevent collisions.
 */
async function initializeLinguisticGatekeeper(rootNode) {
    uniquenessIndex = new SymSpellIndex(2); // Distance of 2 (e.g., Pikabu -> Pikachu)

    const walk = (node) => {
        // Only index valid, searchable creature names
        if (node.word && !node.unsearchable) uniquenessIndex.indexWord(node.word);
        if (node.children) node.children.forEach(walk);
    };
    walk(rootNode);
    console.log("Uniqueness Index Primed with existing creatures.");
}

/**
 * The "Master" Generator: Combines Portmanteaus, SymSpell, and Bitap.
 */
export function generateFilteredNames(s1, s2) {
    // 1. Generate a large batch of raw linguistic candidates
    const rawCandidates = generatePortmanteaus(s1, s2, 100);
    const filteredResults = [];

    for (const cand of rawCandidates) {
        // A. THE UNIQUENESS TEST (SymSpell)
        // Does this collision exist in our existing creatures database?
        const collisions = uniquenessIndex.lookup(cand.name);
        if (collisions.length > 0) {
            // Log it for the demo so we can see the gatekeeper working
            console.log(`Rejecting "${cand.name}" (Too close to ${collisions.join(', ')})`);
            continue;
        }

        // B. THE RECOGNIZABILITY TEST (Bitap)
        // Find best edit distance of parent names within the generated word
        const d1 = BitapScanner.search(cand.name.toLowerCase(), s1.toLowerCase(), 3);
        const d2 = BitapScanner.search(cand.name.toLowerCase(), s2.toLowerCase(), 3);

        cand.recognizability = (d1 + d2) / 2;

        // C. Final Fitness Logic
        // If the parents are too mangled, we penalize the score significantly
        if (cand.recognizability > 2) {
            cand.score -= 20;
        }

        filteredResults.push(cand);
    }

    // Sort by final filtered score and take the top 10
    return filteredResults.sort((a, b) => b.score - a.score).slice(0, 10);
}

/**
 * Page Setup
 */
async function setupPage() {
    console.log("Initializing Creature Generation Suite...");

    const response = await fetch('./pokemon_type_concepts/creatures.json');
    const rootNode = await response.json();

    // 1. Train engines on the creature data
    initLinguisticEngine(rootNode);
    await initializeLinguisticGatekeeper(rootNode);

    // 2. Demo the Pipeline
    const testCases = [
        ["electric", "mouse"],
        ["ghost", "steel"],
        ["seed", "pup"],
        ["sprout", "dog"],
        ["tree", "wolf"],
        ["fire", "bunny"],
        ["flamable", "rabbit"],
        ["bonfire", "hare"],
        ["drip", "seal"],
        ["drop", "sealion"],
        ["deluge", "walrus"],
        ["bulb", "dinosaur"],
        ["char", "salamander"],
        ["scorch", "bunny"],
        ["lit", "kitten"],
        ["incinerate", "roar"],
    ];

    testCases.forEach(([p1, p2]) => {
        console.log(`\n--- Generating for: ${p1} + ${p2} ---`);
        const results = generateFilteredNames(p1, p2);
        results.forEach((res, i) => {
            console.log(`${i + 1}. ${res.name} (Score: ${res.score.toFixed(2)}, Recog: ${res.recognizability})`);
        });
    });
}

window.addEventListener('load', setupPage);