/**
 * A bit-for-bit logical port of thefuzz (Python) to JavaScript.
 */
export class FuzzyMatcher {
    static debug = false;

    static #fullProcess(s) {
        if (!s) return "";
        return s.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').trim().replace(/\s+/g, ' ');
    }

    static #getTokens(s) {
        return this.#fullProcess(s).split(/\s+/).filter(t => t.length > 0);
    }

    static #lcsLength(s1, s2) {
        const m = s1.length;
        const n = s2.length;

        const dp = new Array(m + 1);
        for (let i = 0; i <= m; i++) {
            dp[i] = new Uint16Array(n + 1);
        }

        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (s1[i - 1] === s2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp[m][n];
    }

    static ratio(s1, s2) {
        if (!s1 || !s2) return 0;
        if (s1 === s2) return 100;

        const lenSum = s1.length + s2.length;
        if (lenSum === 0) return 100;

        const matches = this.#lcsLength(s1, s2);

        return Math.round((2 * matches * 100) / lenSum);
    }

    static partialRatio(s1, s2) {
        if (!s1 || !s2) return 0;
        if (s1 === s2) return 100;

        let shorter = s1.length <= s2.length ? s1 : s2;
        let longer = s1.length <= s2.length ? s2 : s1;

        const m = shorter.length;
        const n = longer.length;

        if (m === 0) return 0;

        let maxRatio = 0;

        for (let i = 0; i <= n - m; i++) {
            const window = longer.substring(i, i + m);
            const score = this.ratio(shorter, window);
            if (score > maxRatio) {
                maxRatio = score;
                if (maxRatio === 100) break;
            }
        }

        return maxRatio;
    }

    static tokenSortRatio(s1, s2) {
        const t1 = this.#getTokens(s1).sort().join(' ');
        const t2 = this.#getTokens(s2).sort().join(' ');
        const score = this.ratio(t1, t2);
        if (this.debug) console.log(`[SortRatio] "${t1}" vs "${t2}" = ${score}`);
        return score;
    }

    static partialTokenSortRatio(s1, s2) {
        const t1 = this.#getTokens(s1).sort().join(' ');
        const t2 = this.#getTokens(s2).sort().join(' ');
        const score = this.partialRatio(t1, t2);
        if (this.debug) console.log(`[PartialSort] "${t1}" vs "${t2}" = ${score}`);
        return score;
    }

    static tokenSetRatio(s1, s2) {
        const tokens1 = new Set(this.#getTokens(s1));
        const tokens2 = new Set(this.#getTokens(s2));

        const intersection = [...tokens1].filter(x => tokens2.has(x)).sort().join(' ');
        const diff1to2 = [...tokens1].filter(x => !tokens2.has(x)).sort().join(' ');
        const diff2to1 = [...tokens2].filter(x => !tokens1.has(x)).sort().join(' ');

        const t0 = intersection.trim();
        const t1 = (intersection + " " + diff1to2).trim();
        const t2 = (intersection + " " + diff2to1).trim();

        return Math.max(this.ratio(t0, t1), this.ratio(t0, t2), this.ratio(t1, t2));
    }

    static partialTokenSetRatio(s1, s2) {
        const tokens1 = new Set(this.#getTokens(s1));
        const tokens2 = new Set(this.#getTokens(s2));

        const intersection = [...tokens1].filter(x => tokens2.has(x)).sort().join(' ');
        const diff1to2 = [...tokens1].filter(x => !tokens2.has(x)).sort().join(' ');
        const diff2to1 = [...tokens2].filter(x => !tokens1.has(x)).sort().join(' ');

        const t0 = intersection.trim();
        const t1 = (intersection + " " + diff1to2).trim();
        const t2 = (intersection + " " + diff2to1).trim();

        return Math.max(
            this.partialRatio(t0, t1),
            this.partialRatio(t0, t2),
            this.partialRatio(t1, t2)
        );
    }

    static WRatio(s1, s2) {
        if (!s1 || !s2) return 0;

        const p1 = this.#fullProcess(s1);
        const p2 = this.#fullProcess(s2);

        if (!p1 || !p2) return 0;

        const len1 = p1.length;
        const len2 = p2.length;

        const base = this.ratio(p1, p2);

        const lenRatio = Math.max(len1, len2) / Math.min(len1, len2);
        const tryPartial = lenRatio > 1.5;

        let scores = [base];

        if (tryPartial) {
            const partialScale = lenRatio > 8 ? 0.6 : 0.9;

            scores.push(this.partialRatio(p1, p2) * partialScale);
            scores.push(this.partialTokenSortRatio(p1, p2) * partialScale * 0.95);
            scores.push(this.partialTokenSetRatio(p1, p2) * partialScale * 0.95);
        } else {
            scores.push(this.tokenSortRatio(p1, p2) * 0.95);
            scores.push(this.tokenSetRatio(p1, p2) * 0.95);
        }

        return Math.round(Math.max(...scores));
    }

    static extract(query, choices, {
        limit = 5,
        scorer = this.WRatio
    } = {}) {
        const results = choices.map(choice => [choice, scorer.call(this, query, choice)]);
        return results.sort((a, b) => b[1] - a[1]).slice(0, limit);
    }

    static extractOne(query, choices, options = {}) {
        return this.extract(query, choices, {
            ...options,
            limit: 1
        })[0];
    }
}

/**
 * Symmetric Delete Spelling Correction (SymSpell) 
 * Optimized for O(1) lookups against large dictionaries.
 */
export class SymSpellIndex {
    constructor(maxDistance = 2) {
        this.maxDistance = maxDistance;
        this.deletes = new Map(); // Map<DeleteVariant, Set<OriginalWord>>
        this.words = new Set();
    }

    /**
     * Pre-calculates deletes for a word and adds them to the index.
     */
    indexWord(word) {
        word = word.toLowerCase();
        if (this.words.has(word)) return;
        this.words.add(word);

        const variants = this._getDeletes(word);
        for (const variant of variants) {
            if (!this.deletes.has(variant)) {
                this.deletes.set(variant, new Set());
            }
            this.deletes.get(variant).add(word);
        }
    }

    /**
     * Recursively generates all possible string deletions within maxDistance.
     */
    _getDeletes(word) {
        const queue = [word];
        const results = new Set([word]);

        for (let d = 0; d < this.maxDistance; d++) {
            const nextQueue = [];
            for (const item of queue) {
                if (item.length > 1) {
                    for (let i = 0; i < item.length; i++) {
                        const del = item.slice(0, i) + item.slice(i + 1);
                        if (!results.has(del)) {
                            results.add(del);
                            nextQueue.push(del);
                        }
                    }
                }
            }
            queue.push(...nextQueue);
        }
        return results;
    }

    /**
     * Checks if a word (or something very close to it) exists in the index.
     * Returns an array of matches.
     */
    lookup(word) {
        word = word.toLowerCase();
        const candidates = this._getDeletes(word);
        const matches = new Set();

        for (const variant of candidates) {
            if (this.deletes.has(variant)) {
                this.deletes.get(variant).forEach(orig => matches.add(orig));
            }
        }
        return Array.from(matches);
    }
}

/**
 * Bitap Algorithm (Baeza-Yates–Gonnet)
 * Best for searching a pattern within a longer string using bitwise operations.
 */
export class BitapScanner {
    /**
     * Returns the best edit distance of 'pattern' found anywhere in 'text'.
     * @param {string} text - The longer string to search within.
     * @param {string} pattern - The short pattern to find.
     * @param {number} maxDistance - Max allowed edits.
     */
    static search(text, pattern, maxDistance = 2) {
        const m = pattern.length;
        if (m === 0) return 0;
        if (m > 31) throw new Error("Pattern too long for standard Bitap (max 31 chars)");

        // 1. Precompute bitmasks for each character in the pattern
        const charMasks = {};
        for (let i = 0; i < m; i++) {
            const char = pattern[i];
            charMasks[char] = (charMasks[char] || 0) | (1 << i);
        }

        // 2. Initialize the state bitmasks for each possible edit distance (0 to k)
        // In Bitap, 0 means match, 1 means mismatch
        const R = new Uint32Array(maxDistance + 1);
        for (let i = 0; i <= maxDistance; i++) {
            R[i] = ~1; // All bits 1, except first bit is 0
        }

        let bestDistance = m;

        // 3. Scan the text
        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            const charMask = charMasks[char] || 0;

            const oldR = new Uint32Array(R);

            // Update R[0] (Exact match state)
            R[0] |= ~charMask;
            R[0] <<= 1;

            // Update R[1..k] (Fuzzy match states)
            for (let k = 1; k <= maxDistance; k++) {
                const substitution = (oldR[k - 1] | (~charMask)) << 1;
                const insertion = oldR[k - 1] << 1;
                const deletion = (R[k - 1] | (~charMask)) << 1;

                R[k] = (oldR[k] | (~charMask)) << 1;
                R[k] &= (substitution & insertion & deletion);
            }

            // Check if any state hit a match (the m-th bit is 0)
            for (let k = 0; k <= maxDistance; k++) {
                if (!(R[k] & (1 << m))) {
                    bestDistance = Math.min(bestDistance, k);
                }
            }
        }

        return bestDistance;
    }
}