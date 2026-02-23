import {
    initLinguisticEngine,
    generatePortmanteaus
} from './portmanteau.js';

async function setupPage() {
    const response = await fetch('./pokemon_type_concepts/creatures.json');
    const rootNode = await response.json();

    // 1. Train the linguistic engine on your data
    initLinguisticEngine(rootNode);

    // 2. Example: Generate a name for a Fire + Rabbit concept
    console.log("Portmanteau Results:",  generatePortmanteaus("seed", "pup"));
    console.log("Portmanteau Results:",  generatePortmanteaus("sprout", "dog"));
    console.log("Portmanteau Results:",  generatePortmanteaus("tree", "wolf"));

    console.log("Portmanteau Results:",  generatePortmanteaus("fire", "bunny"));
    console.log("Portmanteau Results:",  generatePortmanteaus("flamable", "rabbit"));
    console.log("Portmanteau Results:",  generatePortmanteaus("bonfire", "hare"));

    console.log("Portmanteau Results:",  generatePortmanteaus("drip", "seal"));
    console.log("Portmanteau Results:",  generatePortmanteaus("drop", "sealion"));
    console.log("Portmanteau Results:",  generatePortmanteaus("deluge", "walrus"));

    console.log("Portmanteau Results:",  generatePortmanteaus("bulb", "dinosaur"));
    console.log("Portmanteau Results:",  generatePortmanteaus("char", "salamander"));
    console.log("Portmanteau Results:",  generatePortmanteaus("scorch", "bunny"));
    console.log("Portmanteau Results:",  generatePortmanteaus("lit", "kitten"));
    console.log("Portmanteau Results:",  generatePortmanteaus("incinerate", "roar"));
}

window.addEventListener('load', setupPage);