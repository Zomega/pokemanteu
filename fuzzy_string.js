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