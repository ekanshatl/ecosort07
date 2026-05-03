import express from "express";
import dotenv from "dotenv";
import * as tf from "@tensorflow/tfjs-node";
import sharp from "sharp";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

let model;

// 🔥 Load model safely
async function loadModel() {
  try {
    // Force CPU backend (important for Render)
    await tf.setBackend("cpu");

    model = await tf.loadLayersModel("file://./model/model.json");

    console.log("✅ Model loaded");
  } catch (err) {
    console.error("❌ Model load error:", err);
    process.exit(1);
  }
}

await loadModel();

// ESP image input
app.use(express.raw({ type: "image/jpeg", limit: "10mb" }));

app.get("/", (req, res) => {
  res.send("🌱 Ecosort ML backend running!");
});

// 🔍 analyze
app.post("/analyze", async (req, res) => {
  try {
    if (!req.body || !req.body.length) {
      return res.status(400).json({ error: "no image" });
    }

    console.log(`🖼️ ${req.body.length} bytes`);

    // 🧠 preprocess
    const img = await sharp(req.body)
      .resize(224, 224)
      .removeAlpha()
      .raw()
      .toBuffer();

    const input = tf.tensor3d(img, [224, 224, 3])
      .expandDims(0)
      .div(255.0);

    const pred = model.predict(input);
    const data = await pred.data();

    const labels = [
      "biodegradable",
      "non_biodegradable",
      "hazardous"
    ];

    let maxIndex = data.indexOf(Math.max(...data));
    const confidence = data[maxIndex];
    const resultClass = labels[maxIndex];

    console.log("📊", data);

    // 🛑 confidence filter
    if (confidence < 0.6) {
      return res.json({
        ok: true,
        result: { class: "unknown", confidence }
      });
    }

    res.json({
      ok: true,
      result: { class: resultClass, confidence }
    });

  } catch (err) {
    console.error("💥 Error:", err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Running on ${PORT}`);
});
