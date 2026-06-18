#!/usr/bin/env node
/**
 * Minimal RAG document Q&A using the Gemini API.
 * No npm installs needed (Node 18+ has built-in fetch).
 *
 * Setup:
 *   export GEMINI_API_KEY=your_key_here
 *   node rag.js path/to/document.txt
 *
 * Then type questions at the "Q:" prompt. Type "exit" to quit.
 */

const fs = require('fs');
const readline = require('readline');

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
  console.error('Set GEMINI_API_KEY environment variable first.');
  process.exit(1);
}

const EMBED_MODEL = 'gemini-embedding-001';
const GEN_MODEL = 'gemini-3.5-flash'; // swap to 'gemini-2.5-flash' if this model isn't available on your key
const BASE = 'https://generativelanguage.googleapis.com/v1beta/models';

const docPath = process.argv[2];
if (!docPath) {
  console.error('Usage: node rag.js <path-to-document.txt>');
  process.exit(1);
}

// ---------- 1. Load & chunk the document ----------

function chunkText(text, chunkSizeWords = 700, overlapWords = 120) {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks = [];
  let i = 0;
  while (i < words.length) {
    chunks.push(words.slice(i, i + chunkSizeWords).join(' '));
    i += chunkSizeWords - overlapWords;
  }
  return chunks;
}

// ---------- 2. Embeddings ----------

async function embedChunks(chunks) {
  const requests = chunks.map(text => ({
    model: `models/${EMBED_MODEL}`,
    content: { parts: [{ text }] },
    taskType: 'RETRIEVAL_DOCUMENT'
  }));

  const res = await fetch(`${BASE}/${EMBED_MODEL}:batchEmbedContents?key=${API_KEY}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ requests })
  });
  const data = await res.json();
  if (!data.embeddings) throw new Error('Embedding failed: ' + JSON.stringify(data));
  return data.embeddings.map(e => e.values);
}

async function embedQuery(text) {
  const res = await fetch(`${BASE}/${EMBED_MODEL}:embedContent?key=${API_KEY}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      content: { parts: [{ text }] },
      taskType: 'RETRIEVAL_QUERY'
    })
  });
  const data = await res.json();
  if (!data.embedding) throw new Error('Query embedding failed: ' + JSON.stringify(data));
  return data.embedding.values;
}

// ---------- 3. Retrieval (cosine similarity, in-memory) ----------

function cosineSimilarity(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

function topKChunks(queryEmbedding, chunkEmbeddings, chunks, k = 4) {
  const scored = chunkEmbeddings.map((emb, i) => ({
    text: chunks[i],
    score: cosineSimilarity(queryEmbedding, emb)
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

// ---------- 4. Generation (augmented prompt) ----------

async function askGemini(question, contextChunks) {
  const context = contextChunks
    .map((c, i) => `[Excerpt ${i + 1}]\n${c.text}`)
    .join('\n\n');

  const prompt = `You are a helpful assistant answering questions about a document. Use ONLY the excerpts below to answer. If the answer isn't in the excerpts, say you don't know.

${context}

Question: ${question}

Answer:`;

  const res = await fetch(`${BASE}/${GEN_MODEL}:generateContent?key=${API_KEY}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] })
  });
  const data = await res.json();
  const answer = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!answer) throw new Error('Generation failed: ' + JSON.stringify(data));
  return answer;
}

// ---------- 5. Main: load doc, embed once, then loop on questions ----------

async function main() {
  console.log('Reading document...');
  const text = fs.readFileSync(docPath, 'utf-8');
  const chunks = chunkText(text);
  console.log(`Split into ${chunks.length} chunks. Embedding...`);

  let chunkEmbeddings = [];
  const BATCH = 90; // stay under per-request limits
  for (let i = 0; i < chunks.length; i += BATCH) {
    const slice = chunks.slice(i, i + BATCH);
    const embs = await embedChunks(slice);
    chunkEmbeddings = chunkEmbeddings.concat(embs);
  }

  console.log('Ready! Ask questions about the document (type "exit" to quit).\n');

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const ask = () => rl.question('Q: ', async (question) => {
    if (question.trim().toLowerCase() === 'exit') {
      rl.close();
      return;
    }
    try {
      const qEmbedding = await embedQuery(question);
      const top = topKChunks(qEmbedding, chunkEmbeddings, chunks, 4);
      const answer = await askGemini(question, top);
      console.log(`\nA: ${answer}\n`);
    } catch (err) {
      console.error('Error:', err.message);
    }
    ask();
  });
  ask();
}

main();