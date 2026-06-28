// drive-and-learn Cloudflare Worker
// Endpoints: GET /topics, POST /search, POST /answer, POST /plan, POST /progress, POST /speak
// Env: OPENAI_API_KEY, GEMINI_API_KEY, NEON_URL (Neon HTTP API URL)

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
};

const EMBED_MODEL = "text-embedding-3-small";
const GEMINI_MODEL = "gemini-2.5-flash";
const GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts";

// Podcast host names — must match the speaker labels the model is told to use.
const PODCAST_HOSTS = { a: "Alex", b: "Sam" };
const VOICES = { single: "Kore", podcastA: "Kore", podcastB: "Puck" };

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });
}

// ---- Neon HTTP API ---------------------------------------------------------
// Neon's SQL-over-HTTP endpoint lives at https://<host>/sql and authenticates
// via the `Neon-Connection-String` header. NEON_URL should be the full Postgres
// connection string (postgresql://user:pass@host/db) that Neon gives you; we
// derive the /sql endpoint from its host. If NEON_URL is already an https .../sql
// URL, set NEON_CONNECTION_STRING separately to carry auth.
// Body is { query, params }; response is { rows: [...] }.
function neonEndpoint(env) {
  const conn = env.NEON_URL || "";
  const headers = { "Content-Type": "application/json" };
  if (/^postgres(ql)?:\/\//i.test(conn)) {
    const u = new URL(conn);
    headers["Neon-Connection-String"] = conn;
    return { endpoint: `https://${u.hostname}/sql`, headers };
  }
  // Already an HTTP(S) endpoint; auth must come from a separate secret.
  if (env.NEON_CONNECTION_STRING) {
    headers["Neon-Connection-String"] = env.NEON_CONNECTION_STRING;
  }
  return { endpoint: conn, headers };
}

async function sql(env, query, params = []) {
  const { endpoint, headers } = neonEndpoint(env);
  const res = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify({ query, params }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Neon query failed (${res.status}): ${text}`);
  }
  const data = await res.json();
  // Neon HTTP returns { rows: [...] }; fall back to raw array if shaped differently.
  return Array.isArray(data) ? data : data.rows || [];
}

// ---- OpenAI embeddings -----------------------------------------------------
async function embed(env, text) {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({ model: EMBED_MODEL, input: text }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`OpenAI embed failed (${res.status}): ${t}`);
  }
  const data = await res.json();
  return data.data[0].embedding;
}

// pgvector wants a string like "[0.1,0.2,...]"
function toVectorLiteral(embedding) {
  return `[${embedding.join(",")}]`;
}

// ---- Gemini Flash ----------------------------------------------------------
async function gemini(env, prompt) {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=` +
    env.GEMINI_API_KEY;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
    }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Gemini failed (${res.status}): ${t}`);
  }
  const data = await res.json();
  const parts = data?.candidates?.[0]?.content?.parts || [];
  return parts.map((p) => p.text || "").join("").trim();
}

// ---- Gemini text-to-speech -------------------------------------------------
// Returns raw PCM (24kHz, 16-bit, mono) as a Uint8Array. `multi` switches to a
// two-host (Alex/Sam) voice config for podcast scripts.
async function geminiTTS(env, text, multi) {
  const speechConfig = multi
    ? {
        multiSpeakerVoiceConfig: {
          speakerVoiceConfigs: [
            {
              speaker: PODCAST_HOSTS.a,
              voiceConfig: { prebuiltVoiceConfig: { voiceName: VOICES.podcastA } },
            },
            {
              speaker: PODCAST_HOSTS.b,
              voiceConfig: { prebuiltVoiceConfig: { voiceName: VOICES.podcastB } },
            },
          ],
        },
      }
    : {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: VOICES.single } },
      };

  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_TTS_MODEL}:generateContent?key=` +
    env.GEMINI_API_KEY;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text }] }],
      generationConfig: { responseModalities: ["AUDIO"], speechConfig },
    }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Gemini TTS failed (${res.status}): ${t}`);
  }
  const data = await res.json();
  const b64 = data?.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!b64) throw new Error("Gemini TTS returned no audio");
  return base64ToBytes(b64);
}

function base64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

// Wrap raw PCM in a 44-byte WAV header so browsers can play it directly.
function pcmToWav(pcm, sampleRate = 24000, channels = 1, bitsPerSample = 16) {
  const blockAlign = (channels * bitsPerSample) / 8;
  const byteRate = sampleRate * blockAlign;
  const dataSize = pcm.length;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // audio format = PCM
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);
  new Uint8Array(buffer, 44).set(pcm);
  return new Uint8Array(buffer);
}

// ---- Semantic search -------------------------------------------------------
async function searchChunks(env, query, matchCount = 5) {
  const embedding = await embed(env, query);
  const vec = toVectorLiteral(embedding);
  const rows = await sql(
    env,
    `SELECT cc.id,
            cc.topic_id,
            t.title          AS topic_title,
            cc.source_type,
            cc.source_title,
            cc.source_url,
            cc.summary,
            cc.raw_text,
            1 - (cc.embedding <=> $1::vector) AS similarity
       FROM content_chunks cc
       LEFT JOIN topics t ON t.id = cc.topic_id
      WHERE cc.embedding IS NOT NULL
      ORDER BY cc.embedding <=> $1::vector
      LIMIT $2`,
    [vec, matchCount]
  );
  return rows;
}

// ---- Intent detection ------------------------------------------------------
function detectIntent(query) {
  const q = (query || "").toLowerCase();
  if (/\b(brief|short|quick)\b/.test(q)) return "brief";
  if (/\b(podcast|deep dive)\b/.test(q)) return "podcast";
  if (/study plan|what should i study/.test(q)) return "plan";
  return "deep";
}

const STYLE_PROMPTS = {
  brief:
    "Answer in 2-3 concise sentences. This is for a driver listening hands-free, so be direct and skip preamble.",
  deep:
    "Give a clear, well-structured spoken explanation that takes about 1-2 minutes to read aloud. Use plain language suitable for hands-free listening while driving. Avoid markdown, code blocks, or bullet symbols.",
  podcast:
    `Write this as an engaging ~5 minute two-host podcast dialogue between ${PODCAST_HOSTS.a} and ${PODCAST_HOSTS.b}. ` +
    `${PODCAST_HOSTS.a} is the curious co-host asking sharp questions; ${PODCAST_HOSTS.b} is the expert who explains clearly. ` +
    `Cover what the concept is, why it matters, real-world examples, and trade-offs. ` +
    `Prefix every line of dialogue with the speaker's name and a colon, exactly like "${PODCAST_HOSTS.a}:" or "${PODCAST_HOSTS.b}:". ` +
    `Alternate naturally between the two. No markdown, no stage directions, just the spoken lines.`,
};

// ---- Build an adaptive 7-day study plan ------------------------------------
async function buildPlan(env, userId = "default") {
  const topics = await sql(
    env,
    `SELECT t.id,
            t.title,
            t.category,
            t.subtitle,
            t.difficulty,
            COALESCE(up.status, 'not_started') AS status
       FROM topics t
       LEFT JOIN user_progress up
         ON up.topic_id = t.id AND up.user_id = $1
      ORDER BY t.difficulty, t.id`,
    [userId]
  );

  const completed = topics.filter((t) => t.status === "completed");
  const remaining = topics.filter((t) => t.status !== "completed");

  const prompt = `You are a system design driving tutor building an adaptive 7-day study plan.

The learner studies hands-free while driving. Here is their progress.

Completed (${completed.length}): ${completed.map((t) => t.title).join(", ") || "none"}

Not yet completed (${remaining.length}):
${remaining
  .map((t) => `- ${t.title} (${t.category}, difficulty ${t.difficulty}): ${t.subtitle || ""}`)
  .join("\n")}

Create a focused 7-day plan that prioritizes the not-yet-completed topics, builds from foundations to advanced, and groups related topics. For each day give: the day number, 1-3 topic titles to cover, and one sentence on why. Write it as plain spoken prose suitable for listening while driving. No markdown.`;

  const plan = await gemini(env, prompt);
  return { plan, completedCount: completed.length, remainingCount: remaining.length };
}

// ---- Route handlers --------------------------------------------------------
async function handleTopics(env) {
  const rows = await sql(
    env,
    `SELECT t.id,
            t.title,
            t.category,
            t.subtitle,
            t.difficulty,
            COALESCE(up.status, 'not_started') AS status,
            up.completed_at
       FROM topics t
       LEFT JOIN user_progress up
         ON up.topic_id = t.id AND up.user_id = $1
      ORDER BY t.id`,
    ["default"]
  );
  return json({ topics: rows });
}

async function handleSearch(env, body) {
  const query = body.query || body.q || "";
  if (!query) return json({ error: "query is required" }, 400);
  const matchCount = body.match_count || 5;
  const results = await searchChunks(env, query, matchCount);
  return json({ query, results });
}

async function handleAnswer(env, body) {
  const query = body.query || body.q || "";
  if (!query) return json({ error: "query is required" }, 400);
  const userId = body.user_id || "default";
  const intent = detectIntent(query);

  // "what should I study" routes into the study plan
  if (intent === "plan") {
    const { plan } = await buildPlan(env, userId);
    return json({ intent: "plan", query, answer: plan });
  }

  const chunks = await searchChunks(env, query, 5);
  const context = chunks
    .map(
      (c, i) =>
        `[Source ${i + 1}: ${c.source_title || c.topic_title || "unknown"}]\n${
          c.summary ? c.summary + "\n" : ""
        }${c.raw_text}`
    )
    .join("\n\n");

  const style = STYLE_PROMPTS[intent] || STYLE_PROMPTS.deep;
  const prompt = `You are a system design driving tutor answering a learner's spoken question using excerpts from technical books and engineering blogs.

Question: ${query}

${style}

Base your answer on the source material below. If the sources do not cover something, rely on your own knowledge but stay accurate.

Source material:
${context || "(no relevant excerpts found)"}`;

  const answer = await gemini(env, prompt);
  return json({
    intent,
    query,
    answer,
    sources: chunks.map((c) => ({
      title: c.source_title,
      topic: c.topic_title,
      url: c.source_url,
      similarity: c.similarity,
    })),
  });
}

async function handlePlan(env, body) {
  const userId = (body && body.user_id) || "default";
  const result = await buildPlan(env, userId);
  return json(result);
}

async function handleProgress(env, body) {
  const topicId = body.topic_id;
  if (!topicId) return json({ error: "topic_id is required" }, 400);
  const userId = body.user_id || "default";
  const status = body.status || "completed";

  const rows = await sql(
    env,
    `INSERT INTO user_progress (user_id, topic_id, status, completed_at, updated_at)
     VALUES ($1, $2, $3, CASE WHEN $3 = 'completed' THEN NOW() ELSE NULL END, NOW())
     ON CONFLICT (user_id, topic_id)
     DO UPDATE SET status = EXCLUDED.status,
                   completed_at = CASE WHEN EXCLUDED.status = 'completed' THEN NOW()
                                       ELSE user_progress.completed_at END,
                   updated_at = NOW()
     RETURNING id, user_id, topic_id, status, completed_at`,
    [userId, topicId, status]
  );
  return json({ progress: rows[0] || null });
}

// High-quality speech: turn answer text into WAV audio via Gemini TTS.
// mode === "podcast" uses two-host multi-speaker voices; everything else is
// a single natural voice. Returns binary audio/wav.
async function handleSpeak(env, body) {
  const text = (body.text || "").trim();
  if (!text) return json({ error: "text is required" }, 400);
  const multi = body.mode === "podcast";
  const pcm = await geminiTTS(env, text, multi);
  const wav = pcmToWav(pcm);
  return new Response(wav, {
    headers: { "Content-Type": "audio/wav", "Cache-Control": "no-store", ...CORS },
  });
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS });
    }

    const url = new URL(request.url);
    const path = url.pathname.replace(/\/$/, "") || "/";

    try {
      if (request.method === "GET" && path === "/topics") {
        return await handleTopics(env);
      }

      if (request.method === "POST") {
        const body = await request.json().catch(() => ({}));
        switch (path) {
          case "/search":
            return await handleSearch(env, body);
          case "/answer":
            return await handleAnswer(env, body);
          case "/plan":
            return await handlePlan(env, body);
          case "/progress":
            return await handleProgress(env, body);
          case "/speak":
            return await handleSpeak(env, body);
        }
      }

      return json({ error: "Not found", path }, 404);
    } catch (err) {
      return json({ error: err.message || String(err) }, 500);
    }
  },
};
