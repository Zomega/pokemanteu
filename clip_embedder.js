import * as tf from 'https://esm.run/@tensorflow/tfjs';

export class ClipEmbedder {
  /**
   * @param {string} textModelUrl - Path to the TFJS text model
   * @param {string} imageModelUrl - Path to the TFJS image model
   */
  constructor(
    textModelUrl = './clip-text-vit-32-tfjs/model.json',
    imageModelUrl = './clip-image-vit-32-tfjs/model.json'
  ) {
    this.textModelUrl = textModelUrl;
    this.imageModelUrl = imageModelUrl;
    this.textModel = null;
    this.imageModel = null;
    this.tokenizer = null;
  }

  async initialize() {
    if (this.textModel && this.imageModel && this.tokenizer) return;

    console.log("Loading CLIP BPE Tokenizer...");
    // TODO: DON'T LOVE THIS EXTERNAL IMPORT...
    const TokenizerModule = await import("https://deno.land/x/clip_bpe@v0.0.6/mod.js");
    const Tokenizer = TokenizerModule.default;
    this.tokenizer = new Tokenizer();

    console.log("Loading CLIP TFJS Text Model...");
    this.textModel = await tf.loadGraphModel(this.textModelUrl);
    
    console.log("Loading CLIP TFJS Image Model...");
    this.imageModel = await tf.loadGraphModel(this.imageModelUrl);
    
    console.log("Both CLIP Models loaded successfully.");
  }

  /**
   * Generates a 512-dimensional embedding vector for a given string of text.
   */
  async embedText(text) {
    if (!this.textModel || !this.tokenizer) throw new Error("Initialize first.");

    let textTokens = this.tokenizer.encodeForCLIP(text);
    const tokenArray = Int32Array.from(textTokens);
    
    const inputTensor = tf.tensor2d(tokenArray, [1, 77], "int32");
    const results = this.textModel.execute({ 'input': inputTensor }, "output");

    const outputTensor = Array.isArray(results) ? results[0] : results;
    const data = await outputTensor.data();

    inputTensor.dispose();
    if (Array.isArray(results)) results.forEach(t => t.dispose());
    else results.dispose();

    return data;
  }

  /**
   * Generates a 512-dimensional embedding vector for an HTML Image/Canvas.
   * @param {HTMLImageElement | HTMLCanvasElement} imageElement
   */
  async embedImage(imageElement) {
    if (!this.imageModel) throw new Error("Initialize first.");

    const inputTensor = tf.tidy(() => {
      // 1. Extract raw pixels -> [Height, Width, 3]
      let img = tf.browser.fromPixels(imageElement).toFloat();

      // 2. Resize to 224x224
      img = tf.image.resizeBilinear(img, [224, 224]);

      // 3. Normalize [0, 255] -> [0, 1]
      img = img.div(tf.scalar(255.0));

      // 4. Normalize using CLIP Mean and Std
      const mean = tf.tensor1d([0.48145466, 0.4578275, 0.40821073]);
      const std = tf.tensor1d([0.26862954, 0.26130258, 0.27577711]);
      // Note: Broad-casting works automatically here
      img = img.sub(mean).div(std);

      // --- STEP 5 REMOVED ---
      // Do NOT transpose to [3, 224, 224]. 
      // Keep it as [224, 224, 3].

      // 6. Add the Batch dimension -> [1, 224, 224, 3]
      return img.expandDims(0);
    });

    // Execute the image model
    const results = this.imageModel.execute({ 'input': inputTensor }, "output");
    const outputTensor = Array.isArray(results) ? results[0] : results;
    const data = await outputTensor.data();

    // Cleanup
    inputTensor.dispose();
    if (Array.isArray(results)) results.forEach(t => t.dispose());
    else results.dispose();

    return data;
  }
}