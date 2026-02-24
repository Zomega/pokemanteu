import {
    describe,
    it,
    expect,
    beforeAll
} from 'vitest';
import {
    FuzzyMatcher as fuzz,
    BitapScanner,
    SymSpellIndex
} from './fuzzy_string.js';

describe('FuzzyMatcher - Python Baseline Alignment', () => {
    it('Simple Ratio should handle punctuation correctly', () => {
        expect(fuzz.ratio("this is a test", "this is a test!")).toBe(97);
    });

    it('Partial Ratio should find perfect substrings', () => {
        expect(fuzz.partialRatio("this is a test", "this is a test!")).toBe(100);
    });

    it('Token Sort Ratio should ignore word order', () => {
        const s1 = "fuzzy wuzzy was a bear";
        const s2 = "wuzzy fuzzy was a bear";
        expect(fuzz.ratio(s1, s2)).toBe(91);
        expect(fuzz.tokenSortRatio(s1, s2)).toBe(100);
    });

    it('Token Set Ratio should handle duplicate words', () => {
        const s1 = "fuzzy was a bear";
        const s2 = "fuzzy fuzzy was a bear";
        expect(fuzz.tokenSetRatio(s1, s2)).toBe(100);
    });

    it('WRatio (Process) should match Dallas Cowboys at 90', () => {
        const choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"];
        const result = fuzz.extractOne("cowboys", choices);
        expect(result[0]).toBe("Dallas Cowboys");
        expect(result[1]).toBe(90);
    });
});

describe('Search Algorithms - SymSpell & Bitap', () => {
    let index;

    beforeAll(() => {
        index = new SymSpellIndex(2);
        index.indexWord("Pikachu");
        index.indexWord("Bulbasaur");
    });

    it('SymSpell should catch near-misses (Uniqueness Test)', () => {
        const matches = index.lookup("Pikabu");
        expect(matches).toContain("pikachu");
    });

    it('Bitap should find patterns inside strings (Recognizability Test)', () => {
        // Looking for "Pika" inside "Pikabunny"
        const dist = BitapScanner.search("pikabunny", "pika", 1);
        expect(dist).toBeLessThanOrEqual(1);
    });

    it('Bitap should reject highly mangled parents', () => {
        // "pika" vs "pkm" (distance too high)
        const dist = BitapScanner.search("pkm", "pika", 2);
        expect(dist).toBeGreaterThan(2);
    });
});