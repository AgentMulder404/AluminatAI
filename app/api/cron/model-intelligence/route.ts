// GET /api/cron/model-intelligence
// Schedule: 0 8 * * 1,4 (Monday and Thursday at 8 AM UTC)
// Scans HuggingFace for new trending models, profiles them using roofline
// heuristics, estimates GPU pairings, and stores in model_registry.
// Requires CRON_SECRET bearer token.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { verifyCronSecret } from "@/lib/auth-helpers";

export const runtime = "edge";

const HF_API = "https://huggingface.co/api";
const MAX_MODELS = 20;
const MIN_DOWNLOADS = 1000;

const ARCHITECTURE_INTENSITY: Record<string, [number, string]> = {
  LlamaForCausalLM: [90, "bf16"],
  MistralForCausalLM: [85, "bf16"],
  MixtralForCausalLM: [105, "bf16"],
  Qwen2ForCausalLM: [85, "bf16"],
  PhiForCausalLM: [80, "bf16"],
  Phi3ForCausalLM: [80, "bf16"],
  GemmaForCausalLM: [85, "bf16"],
  Gemma2ForCausalLM: [85, "bf16"],
  GPTNeoXForCausalLM: [100, "fp16"],
  FalconForCausalLM: [95, "bf16"],
  DeepseekV2ForCausalLM: [110, "bf16"],
  DeepseekV3ForCausalLM: [120, "bf16"],
  BertModel: [10, "fp16"],
  BertForMaskedLM: [10, "fp16"],
  RobertaModel: [12, "fp16"],
  T5ForConditionalGeneration: [95, "bf16"],
  WhisperForConditionalGeneration: [25, "fp16"],
  ViTModel: [35, "fp16"],
  CLIPModel: [30, "fp16"],
  StableDiffusionPipeline: [48, "fp16"],
};

const PIPELINE_TAG_INTENSITY: Record<string, [number, string]> = {
  "text-generation": [100, "bf16"],
  "text2text-generation": [95, "bf16"],
  "fill-mask": [10, "fp16"],
  "text-to-image": [48, "fp16"],
  "image-to-image": [48, "fp16"],
  "automatic-speech-recognition": [25, "fp16"],
  "image-classification": [35, "fp16"],
  "feature-extraction": [12, "fp16"],
};

const SUPPORTED_PIPELINE_TAGS = new Set(Object.keys(PIPELINE_TAG_INTENSITY));

function scaleByParams(base: number, paramCount: number | null): number {
  if (!paramCount || paramCount <= 0) return base;
  if (paramCount < 1e9) return base * 0.85;
  if (paramCount < 7e9) return base;
  if (paramCount < 30e9) return base * 1.15;
  if (paramCount < 70e9) return base * 1.35;
  if (paramCount < 200e9) return base * 1.55;
  return base * 1.7;
}

function normalizeTag(modelId: string): string {
  const parts = modelId.split("/");
  let name = (parts.length > 1 ? parts[parts.length - 1] : parts[0]).toLowerCase();
  name = name.replace(/^meta[-_]/, "");
  name = name.replace(/-+/g, "-").replace(/^-|-$/g, "");
  return name;
}

interface HFModel {
  modelId?: string;
  id?: string;
  pipeline_tag?: string;
  config?: { architectures?: string[] };
  safetensors?: { total?: number };
  downloads?: number;
  trendingScore?: number;
  library_name?: string;
  license?: string;
  createdAt?: string;
}

export async function GET(req: NextRequest) {
  const isAuthed = await verifyCronSecret(req.headers.get("authorization"));
  if (!isAuthed) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();

  // Fetch trending models from HuggingFace
  let hfModels: HFModel[] = [];
  try {
    const resp = await fetch(
      `${HF_API}/models?sort=trending&direction=-1&limit=${MAX_MODELS}`,
      { signal: AbortSignal.timeout(30000) }
    );
    if (resp.ok) {
      hfModels = await resp.json();
    }
  } catch {
    return NextResponse.json(
      { error: "Failed to fetch from HuggingFace" },
      { status: 502 }
    );
  }

  // Filter to supported pipeline tags and minimum downloads
  hfModels = hfModels.filter(
    (m) =>
      m.pipeline_tag &&
      SUPPORTED_PIPELINE_TAGS.has(m.pipeline_tag) &&
      (m.downloads ?? 0) >= MIN_DOWNLOADS
  );

  // Get existing model_ids from registry
  const { data: existing } = await supabase
    .from("model_registry")
    .select("model_id");

  const knownIds = new Set((existing ?? []).map((r) => r.model_id as string));

  // Filter out known models
  const newModels = hfModels.filter((m) => {
    const id = m.modelId || m.id || "";
    return id && !knownIds.has(id);
  });

  let registered = 0;

  for (const model of newModels) {
    const modelId = model.modelId || model.id || "";
    const tag = normalizeTag(modelId);
    const pipelineTag = model.pipeline_tag || "";
    const architectures = model.config?.architectures ?? [];
    const architecture = architectures[0] || "";

    // Profile: resolve math intensity
    let baseIntensity: number;
    let precision: string;

    if (architecture && ARCHITECTURE_INTENSITY[architecture]) {
      [baseIntensity, precision] = ARCHITECTURE_INTENSITY[architecture];
    } else if (PIPELINE_TAG_INTENSITY[pipelineTag]) {
      [baseIntensity, precision] = PIPELINE_TAG_INTENSITY[pipelineTag];
    } else {
      baseIntensity = 60;
      precision = "fp16";
    }

    const paramCount =
      typeof model.safetensors?.total === "number"
        ? model.safetensors.total
        : null;

    const mathIntensity = Math.round(scaleByParams(baseIntensity, paramCount) * 10) / 10;
    const isMemoryBound = mathIntensity < 50;

    // Infer family from architecture or pipeline tag
    let family = "Unknown";
    if (architecture.includes("Llama")) family = "Llama";
    else if (architecture.includes("Mistral") || architecture.includes("Mixtral")) family = "Mistral";
    else if (architecture.includes("Qwen")) family = "Qwen";
    else if (architecture.includes("Phi")) family = "Phi";
    else if (architecture.includes("Gemma")) family = "Gemma";
    else if (architecture.includes("Falcon")) family = "Falcon";
    else if (architecture.includes("Deepseek")) family = "DeepSeek";
    else if (architecture.includes("Bert") || architecture.includes("Roberta")) family = "BERT";
    else if (architecture.includes("T5")) family = "T5";
    else if (architecture.includes("Whisper")) family = "Whisper";
    else if (architecture.includes("ViT") || architecture.includes("CLIP")) family = "ViT";
    else if (architecture.includes("StableDiffusion")) family = "Diffusion";
    else if (pipelineTag === "text-generation") family = "LLM";
    else if (pipelineTag === "text-to-image") family = "Diffusion";

    const row = {
      model_id: modelId,
      tag,
      family,
      source: "huggingface",
      math_intensity: mathIntensity,
      precision,
      is_memory_bound: isMemoryBound,
      typical_util_min: isMemoryBound ? 35 : 55,
      typical_util_max: isMemoryBound ? 65 : 85,
      parameter_count: paramCount,
      architecture,
      library: model.library_name || null,
      license: typeof model.license === "string" ? model.license : null,
      downloads_30d: model.downloads ?? 0,
      trending_score: typeof model.trendingScore === "number" ? model.trendingScore : 0,
      profiled_at: new Date().toISOString(),
      hf_metadata: model,
      status: "profiled",
    };

    const { error } = await supabase.from("model_registry").upsert(row, {
      onConflict: "model_id",
    });

    if (!error) {
      registered++;
    }
  }

  return NextResponse.json({
    scanned: hfModels.length,
    new_models: newModels.length,
    registered,
  });
}
