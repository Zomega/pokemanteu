import {
    FuzzyMatcher as fuzz
} from './fuzzy_string.js';

console.log("--- Simple Ratio ---");
console.log(fuzz.ratio("this is a test", "this is a test!"));
// Expected: 97

console.log("\n--- Partial Ratio ---");
console.log(fuzz.partialRatio("this is a test", "this is a test!"));
// Expected: 100

console.log("\n--- Token Sort Ratio ---");
console.log(fuzz.ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear"));
// Expected: 91
console.log(fuzz.tokenSortRatio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear"));
// Expected: 100

console.log("\n--- Token Set Ratio ---");
console.log(fuzz.tokenSortRatio("fuzzy was a bear", "fuzzy fuzzy was a bear"));
// Expected: 84
console.log(fuzz.tokenSetRatio("fuzzy was a bear", "fuzzy fuzzy was a bear"));
// Expected: 100

console.log("\n--- Partial Token Sort Ratio ---");
console.log(fuzz.tokenSortRatio("fuzzy was a bear", "wuzzy fuzzy was a bear"));
// Expected: 84
console.log(fuzz.partialTokenSortRatio("fuzzy was a bear", "wuzzy fuzzy was a bear"));
// Expected: 100

console.log("\n--- Process ---");
const choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"];
console.log(fuzz.extract("new york jets", choices, {
    limit: 2
}));
// Expected: [["New York Jets", 100], ["New York Giants", 78]]

console.log(fuzz.extractOne("cowboys", choices));
// Expected: ["Dallas Cowboys", 90]

console.log("\n--- Advanced Scorer Matching ---");
const songs = [
    "/music/library/good/System of a Down/2005 - Hypnotize/01 - Attack.mp3",
    "/music/library/good/System of a Down/2005 - Hypnotize/10 - She's Like Heroin.mp3"
];
const query = "System of a down - Hypnotize - Heroin";

console.log("Default Ratio:", fuzz.extractOne(query, songs));
// Matches the "Attack" file via sheer volume of common characters
console.log("Token Sort Ratio:", fuzz.extractOne(query, songs, {
    scorer: fuzz.tokenSortRatio
}));
// Correctly matches "She's Like Heroin.mp3" because of shared key tokens