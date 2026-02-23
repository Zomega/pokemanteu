import {
    ClipEmbedder
} from './clip_embedder.js'; // Note the class name change
import * as tf from 'https://esm.run/@tensorflow/tfjs';

// Global state
let clip, database;

/**
 * Utility function to calculate Cosine Similarity between two embeddings.
 */
export function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function extractDatabase(node, db = []) {
    if (node.word && node.unsearchable !== true && node.vector) {
        db.push({
            word: node.word,
            vector: node.vector
        });
    }
    if (node.children) {
        for (const child of node.children) extractDatabase(child, db);
    }
    return db;
}

async function initializeSearch() {
    const statusEl = document.getElementById('results-body');
    try {
        await tf.setBackend('webgl');
        await tf.ready();

        // Now initializes both models
        clip = new ClipEmbedder();
        await clip.initialize();

        const response = await fetch('./pokemon_type_concepts/creatures.json');
        const rootNode = await response.json();
        database = extractDatabase(rootNode);

        document.getElementById('search-btn').disabled = false;
        document.getElementById('image-upload').disabled = false;
        statusEl.innerHTML = `<tr><td colspan="2" style="text-align: center; color: green;">Multi-modal CLIP ready!</td></tr>`;
    } catch (err) {
        statusEl.innerHTML = `<tr><td colspan="2" style="color: red;">Error: ${err.message}</td></tr>`;
    }
}

/**
 * Shared rendering logic for results
 */
function renderResults(queryVector) {
    const resultsBody = document.getElementById('results-body');

    for (const entry of database) {
        entry.score = cosineSimilarity(queryVector, entry.vector);
    }

    database.sort((a, b) => b.score - a.score);
    const top10 = database.slice(0, 10);

    resultsBody.innerHTML = top10.map(item => `
    <tr>
      <td style="padding: 8px; font-weight: bold;">${item.word}</td>
      <td style="padding: 8px;">
        <div style="background: #eee; width: 100px; height: 10px; display: inline-block; border-radius: 5px;">
            <div style="background: #4caf50; width: ${Math.max(0, item.score) * 100}px; height: 10px; border-radius: 5px;"></div>
        </div>
        ${item.score.toFixed(3)}
      </td>
    </tr>
  `).join('');
}

async function handleTextSearch() {
    const query = document.getElementById('clipQuery').value;
    if (!query) return;
    const vector = await clip.embedText(query);
    renderResults(vector);
}

/**
 * New functionality: Search via Image
 */
async function handleImageSearch(event) {
    const file = event.target.files[0];
    if (!file) return;

    const resultsBody = document.getElementById('results-body');
    resultsBody.innerHTML = `<tr><td colspan="2" style="text-align: center;">Processing pixels...</td></tr>`;

    // Create an invisible image element to load the file
    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        const vector = await clip.embedImage(img);
        renderResults(vector);
        URL.revokeObjectURL(img.src); // Cleanup memory
    };
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('search-btn').addEventListener('click', handleTextSearch);
    document.getElementById('clipQuery').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleTextSearch();
    });

    // Connect the new image input
    const imgInput = document.getElementById('image-upload');
    if (imgInput) imgInput.addEventListener('change', handleImageSearch);

    initializeSearch();
});