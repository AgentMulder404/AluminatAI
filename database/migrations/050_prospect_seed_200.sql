-- Seed 200 researched prospects with reasoning for why they're AluminatAI customers.
-- Source: manual research (2026-05-11). Run in Supabase SQL Editor.

BEGIN;

INSERT INTO prospects (company_name, industry, category, company_size, description, notes)
VALUES

-- ═══════════════════════════════════════════════════════════════
-- CATEGORY 1: GPU Cloud Providers (25)
-- ═══════════════════════════════════════════════════════════════

('CoreWeave', 'GPU Cloud', 'cloud-users', '501-1000',
 'AI hyperscaler operating 100K+ NVIDIA GPUs, serving OpenAI, Mistral, and IBM.',
 'WHY: Needs per-customer cost attribution and utilization tracking at massive scale. Waste detection on idle reserved capacity directly impacts margins.'),

('Lambda', 'GPU Cloud', 'cloud-users', '201-500',
 'GPU cloud provider, just raised $1B to expand gigawatt-scale AI data centers.',
 'WHY: Scaling fast from hundreds to thousands of GPUs — waste detection becomes critical before costs spiral. New data centers need day-1 monitoring.'),

('RunPod', 'GPU Cloud', 'cloud-users', '51-200',
 'Serverless and on-demand GPU cloud for AI/ML workloads.',
 'WHY: Serverless GPU model needs real-time per-job cost metering to bill customers accurately. Idle detection optimizes autoscaling.'),

('Nebius', 'GPU Cloud', 'cloud-users', '1001-5000',
 'AI cloud with 50K+ NVIDIA GPUs across 7 countries (US, EU, Middle East).',
 'WHY: Multi-region fleet needs centralized monitoring dashboard. Carbon tracking per data center for EU sustainability reporting.'),

('TensorDock', 'GPU Cloud', 'cloud-users', '11-50',
 'Bare-metal GPU rental optimized for deep learning frameworks.',
 'WHY: Thin margins on bare-metal rental — maximizing GPU utilization is the difference between profit and loss. Idle detection directly impacts revenue.'),

('Paperspace (DigitalOcean)', 'GPU Cloud', 'cloud-users', '201-500',
 'GPU cloud for ML, now part of DigitalOcean.',
 'WHY: Needs to show customers cost breakdowns per experiment/notebook to reduce churn. Enterprise customers demand usage reports.'),

('Vast.ai', 'GPU Cloud', 'cloud-users', '11-50',
 'Decentralized GPU marketplace connecting GPU owners with renters.',
 'WHY: Distributed fleet of heterogeneous GPUs from multiple owners — cant see utilization without an agent on each machine. Perfect AluminatAI use case.'),

('FluidStack', 'GPU Cloud', 'cloud-users', '11-50',
 'Distributed GPU cloud aggregating capacity from multiple sources.',
 'WHY: Aggregated GPUs from data centers, crypto miners, and cloud — needs unified monitoring across wildly different hardware.'),

('Hyperstack', 'GPU Cloud', 'cloud-users', '51-200',
 'NVIDIA Elite partner GPU cloud for AI training and inference.',
 'WHY: Growing fast as NVIDIA partner — needs operational tooling before fleet management becomes chaotic at scale.'),

('GMI Cloud', 'GPU Cloud', 'cloud-users', '51-200',
 'Budget GPU cloud offering H200 at $2.50/hr.',
 'WHY: Competing on price means margins are razor-thin. Waste detection on idle instances saves their bottom line directly.'),

('Vultr', 'GPU Cloud', 'cloud-users', '201-500',
 'Cloud provider expanding into GPU instances for AI workloads.',
 'WHY: Entering GPU market as a traditional cloud provider — needs GPU monitoring tooling they havent built in-house yet.'),

('Latitude.sh', 'GPU Cloud', 'cloud-users', '51-200',
 'Bare-metal GPU servers for AI/ML workloads.',
 'WHY: Per-customer energy attribution needed for accurate billing. Bare-metal means they cant rely on hypervisor-level metering.'),

('Salad', 'GPU Cloud', 'cloud-users', '51-200',
 'Distributed GPU network leveraging consumer-grade GPUs.',
 'WHY: Heterogeneous consumer GPUs (3060s to 4090s) make standardized monitoring essential. No two nodes are alike.'),

('Shadeform', 'GPU Cloud', 'cloud-users', '11-50',
 'GPU aggregator providing unified access across cloud providers.',
 'WHY: Customers use GPUs across multiple providers — needs unified cost tracking to show true spend vs. per-provider fragmented bills.'),

('Crusoe Energy', 'GPU Cloud', 'data-centers', '201-500',
 'GPU data centers powered by stranded natural gas and renewable energy.',
 'WHY: Carbon tracking IS their brand differentiator. AluminatAI carbon metrics let them prove green claims to customers with real data.'),

('IREN', 'GPU Cloud', 'data-centers', '201-500',
 'GPU clusters for AI in renewable-rich regions (hydro, wind).',
 'WHY: Markets themselves as sustainable AI compute — needs carbon metrics to back up the marketing with verifiable data.'),

('Applied Digital', 'GPU Cloud', 'data-centers', '201-500',
 'Building next-gen AI data centers for hyperscale GPU deployments.',
 'WHY: Scaling from design to thousands of GPUs — needs monitoring infrastructure from day one, not bolted on later.'),

('Nscale', 'GPU Cloud', 'cloud-users', '51-200',
 'European sovereign GPU cloud for AI workloads.',
 'WHY: EU CSRD sustainability reporting requirements make carbon tracking per-workload mandatory for their customers.'),

('DataCrunch', 'GPU Cloud', 'cloud-users', '11-50',
 'Finnish GPU cloud running on Nordic renewable energy.',
 'WHY: Nordic green energy positioning benefits from carbon tracking proof points. Small team means they need turnkey monitoring.'),

('Ori Industries', 'GPU Cloud', 'cloud-users', '51-200',
 'Edge GPU cloud for distributed AI inference.',
 'WHY: Distributed edge nodes need per-site power and utilization monitoring. Cant walk to each node to check status.'),

('Together AI', 'GPU Cloud', 'cloud-users', '51-200',
 'GPU cloud optimized for open-source model training and inference.',
 'WHY: Needs per-model cost attribution to accurately price their inference API. Different models have wildly different GPU costs.'),

('Fireworks AI', 'GPU Cloud', 'cloud-users', '51-200',
 'Fast inference platform for generative AI models.',
 'WHY: Cost-per-token across GPU fleet determines their pricing — needs granular GPU cost tracking to stay competitive.'),

('Modal', 'GPU Cloud', 'cloud-users', '51-200',
 'Serverless GPU compute platform for ML workloads.',
 'WHY: Ephemeral serverless containers make cost attribution harder without proper tooling. Idle containers waste money silently.'),

('Beam', 'GPU Cloud', 'cloud-users', '11-50',
 'Serverless GPU platform for deploying ML models.',
 'WHY: Same serverless attribution challenge — needs to track GPU cost per function invocation for billing accuracy.'),

('Brev.dev', 'GPU Cloud', 'cloud-users', '11-50',
 'GPU dev environments for ML engineers.',
 'WHY: Developers forget to shut down GPU instances. Idle detection catches forgotten dev environments burning $2+/hr.'),

-- ═══════════════════════════════════════════════════════════════
-- CATEGORY 2: AI Foundation Model Companies (25)
-- ═══════════════════════════════════════════════════════════════

('Anthropic', 'AI / LLMs', 'ai-startups', '1001-5000',
 'AI safety company building Claude. Trains frontier models on massive GPU clusters.',
 'WHY: Training runs cost tens of millions. Per-experiment cost attribution across GPU clusters helps optimize spend. Even 1% waste savings = millions.'),

('OpenAI', 'AI / LLMs', 'ai-startups', '5001-10000',
 'Largest AI lab, building GPT models. Biggest GPU consumer globally.',
 'WHY: Operates more GPUs than most countries have servers. 1% waste detection across their fleet saves tens of millions annually.'),

('Mistral AI', 'AI / LLMs', 'ai-startups', '201-500',
 'French AI lab building open-weight frontier models.',
 'WHY: EU regulations will require carbon reporting on training runs. European HQ makes ESG compliance non-optional.'),

('Cohere', 'AI / LLMs', 'ai-startups', '201-500',
 'Enterprise LLM provider offering fine-tuning and RAG.',
 'WHY: Fine-tunes models per customer — needs per-customer GPU cost attribution to price services profitably.'),

('AI21 Labs', 'AI / LLMs', 'ai-startups', '201-500',
 'Israeli AI lab building Jamba models for enterprise.',
 'WHY: Multiple model sizes and experiments — needs to track which training experiments waste compute vs. deliver results.'),

('Stability AI', 'AI / LLMs', 'ai-startups', '201-500',
 'Open-source image and video generation models (Stable Diffusion).',
 'WHY: Financial struggles mean GPU cost discipline is existential. Waste detection prevents burning cash on failed training runs.'),

('Hugging Face', 'AI / LLMs', 'ai-startups', '501-1000',
 'Open-source AI platform hosting 500K+ models. Runs inference endpoints.',
 'WHY: Inference endpoints for thousands of models need per-model cost tracking. Some models are 100x more expensive than others.'),

('Aleph Alpha', 'AI / LLMs', 'ai-startups', '201-500',
 'German sovereign AI company building Luminous models.',
 'WHY: German ESG requirements make carbon tracking per training run essential. Government contracts require cost transparency.'),

('Inflection AI', 'AI / LLMs', 'ai-startups', '201-500',
 'Consumer AI assistant (Pi). Pivoting to enterprise.',
 'WHY: Inference GPU costs scale with user growth. Needs to optimize cost-per-conversation as Pi user base expands.'),

('Character.ai', 'AI / LLMs', 'ai-startups', '201-500',
 'Consumer AI chat platform with millions of daily active users.',
 'WHY: Millions of concurrent conversations means massive inference GPU fleet. Cost-per-chat determines viability.'),

('Perplexity AI', 'AI / LLMs', 'ai-startups', '201-500',
 'AI-powered search engine processing millions of queries daily.',
 'WHY: Every search query hits their GPU fleet. Need cost-per-search metrics to know if their unit economics work.'),

('Reka AI', 'AI / LLMs', 'ai-startups', '51-200',
 'Multimodal AI models (text, image, video understanding).',
 'WHY: Video and image training is extremely GPU-intensive — 10x more than text. Waste detection on stuck video training jobs saves enormous cost.'),

('Adept AI', 'AI / LLMs', 'ai-startups', '51-200',
 'Building AI agents that take actions in software.',
 'WHY: Agent training requires GPU-intensive reinforcement learning. Needs experiment-level cost tracking to iterate efficiently.'),

('Sakana AI', 'AI / LLMs', 'ai-startups', '51-200',
 'Tokyo-based AI lab founded by former Google Brain researchers.',
 'WHY: Japan has some of the highest energy costs globally — GPU efficiency is a top priority for their bottom line.'),

('Poolside AI', 'AI / LLMs', 'ai-startups', '51-200',
 'Code generation AI, raised $500M for frontier code models.',
 'WHY: $500M raised and building on massive GPU clusters — investor pressure means they need detailed compute cost reporting.'),

('Databricks (Mosaic)', 'AI / LLMs', 'ml-teams', '5001-10000',
 'Data + AI platform. Acquired MosaicML for GPU training.',
 'WHY: Runs GPU training for enterprise customers — needs per-customer cost attribution for their managed training service.'),

('DeepSeek', 'AI / LLMs', 'ai-startups', '201-500',
 'Chinese AI lab training frontier models competitive with GPT-4.',
 'WHY: Massive GPU fleet for training models like DeepSeek-V2. Fleet monitoring needed at scale.'),

('xAI', 'AI / LLMs', 'ai-startups', '201-500',
 'Elon Musks AI company. Colossus supercomputer has 150K+ GPUs.',
 'WHY: 150K GPUs in one cluster is unprecedented. Monitoring at this scale requires dedicated tooling — manual tracking is impossible.'),

('Meta FAIR', 'AI / LLMs', 'ml-teams', '10001+',
 'Meta AI research lab with 350K+ H100 GPUs training Llama models.',
 'WHY: Multiple internal research teams sharing GPUs need chargeback. Waste detection across 350K GPUs finds millions in savings.'),

('Google DeepMind', 'AI / LLMs', 'ml-teams', '5001-10000',
 'Googles AI research division, combined DeepMind + Google Brain.',
 'WHY: Though TPU-heavy, GPU workloads for specific research still need monitoring. Multi-team chargeback is complex.'),

('Amazon AGI', 'AI / LLMs', 'ml-teams', '10001+',
 'Amazons foundation model research team building on GPU clusters.',
 'WHY: Internal GPU cluster shared across AGI, Alexa, and AWS teams — needs cost attribution per division.'),

('Apple ML Research', 'AI / LLMs', 'ml-teams', '10001+',
 'Apples growing ML team for on-device AI and Apple Intelligence.',
 'WHY: Rapidly growing GPU footprint for training on-device models. Internal cost tracking needed as spend ramps.'),

('Samsung AI Center', 'AI / LLMs', 'ml-teams', '10001+',
 'Samsung AI research centers across US, UK, Korea, Canada.',
 'WHY: Multiple global research centers sharing GPU resources — needs unified monitoring and per-site cost reporting.'),

('Writer', 'AI / LLMs', 'ai-startups', '201-500',
 'Enterprise AI writing platform with custom LLMs.',
 'WHY: Fine-tunes custom models per enterprise customer — per-tenant GPU cost attribution determines pricing accuracy.'),

('Baidu AI', 'AI / LLMs', 'ml-teams', '10001+',
 'Chinese tech giant training Ernie LLMs on large GPU clusters.',
 'WHY: Large GPU fleet for LLM training — utilization monitoring needed across Chinas most-used AI platform.'),

-- ═══════════════════════════════════════════════════════════════
-- CATEGORY 3: AI Application Startups (30)
-- ═══════════════════════════════════════════════════════════════

('Midjourney', 'AI / Image Generation', 'ai-startups', '51-200',
 'Leading AI image generation platform with millions of users.',
 'WHY: Inference GPU costs are their #1 operating expense. Cost-per-image tracking determines if pricing tiers are profitable.'),

('Runway', 'AI / Video Generation', 'ai-startups', '201-500',
 'AI video generation and editing tools (Gen-3 Alpha).',
 'WHY: Video generation training costs are enormous — 100x more GPU-intensive than text. Per-model cost tracking prevents budget blowouts.'),

('Pika', 'AI / Video Generation', 'ai-startups', '51-200',
 'AI video generation startup competing with Runway.',
 'WHY: GPU-intensive video rendering at scale. Budget alerts catch runaway training jobs before they cost $50K overnight.'),

('ElevenLabs', 'AI / Audio', 'ai-startups', '51-200',
 'AI voice synthesis and cloning platform.',
 'WHY: Real-time voice inference at scale means GPU cost is core to unit economics. Cost-per-minute-of-audio tracking needed.'),

('Synthesia', 'AI / Video', 'ai-startups', '201-500',
 'AI video avatar generation for enterprise communications.',
 'WHY: GPU rendering costs scale linearly with every customer video generated. Need per-video cost tracking for pricing.'),

('Jasper AI', 'AI / Content', 'ai-startups', '201-500',
 'Enterprise AI content generation platform.',
 'WHY: Multiple model fine-tunes for enterprise customers — per-customer GPU cost tracking prevents margin erosion.'),

('Glean', 'AI / Enterprise Search', 'ai-startups', '201-500',
 'Enterprise AI search across company documents and systems.',
 'WHY: Inference costs scale with every enterprise customer and query volume. GPU cost-per-customer determines profitability.'),

('Harvey AI', 'AI / Legal', 'ai-startups', '51-200',
 'AI-powered legal research and document platform.',
 'WHY: Fine-tunes models per law firm — per-customer GPU cost tracking determines if each client is profitable.'),

('Abridge', 'AI / Healthcare', 'ai-startups', '201-500',
 'Medical AI transcription and clinical documentation.',
 'WHY: HIPAA-regulated environment needs auditable GPU cost reporting. Real-time transcription inference at scale.'),

('Hippocratic AI', 'AI / Healthcare', 'ai-startups', '51-200',
 'AI agents for healthcare with safety-critical requirements.',
 'WHY: Safety-critical training needs careful experiment tracking — cost monitoring catches anomalous training runs early.'),

('Inworld AI', 'AI / Gaming', 'ai-startups', '51-200',
 'AI NPCs for games with real-time character behavior.',
 'WHY: Real-time inference for millions of simultaneous game sessions. GPU costs scale with concurrent players.'),

('Replika', 'AI / Consumer', 'ai-startups', '51-200',
 'AI companion chatbot with millions of active users.',
 'WHY: Millions of concurrent conversations means massive inference fleet. Cost-per-conversation determines subscription pricing.'),

('Luma AI', 'AI / 3D', 'ai-startups', '51-200',
 'Neural 3D capture and generation from text/images.',
 'WHY: GPU-intensive neural radiance field training and rendering. Each 3D generation job consumes significant GPU time.'),

('Ideogram', 'AI / Image Generation', 'ai-startups', '51-200',
 'AI image generation with strong text rendering capabilities.',
 'WHY: Competing with Midjourney on GPU-intensive image generation — cost efficiency determines competitive pricing.'),

('Suno AI', 'AI / Music', 'ai-startups', '51-200',
 'AI music generation from text prompts.',
 'WHY: Audio model training involves long GPU-intensive runs on spectrogram data. Waste detection catches stuck audio training.'),

('Udio', 'AI / Music', 'ai-startups', '11-50',
 'AI music generation platform.',
 'WHY: Same GPU-intensive audio training. Small team means they cant afford wasted GPU cycles on a startup budget.'),

('Typeface', 'AI / Enterprise Content', 'ai-startups', '201-500',
 'Enterprise AI for brand-specific content generation.',
 'WHY: Per-brand model fine-tuning for each enterprise customer — needs per-tenant GPU cost attribution for pricing.'),

('Moveworks', 'AI / IT Support', 'ai-startups', '201-500',
 'AI-powered IT support and employee service desk.',
 'WHY: Inference costs scale with every enterprise customer ticket. GPU cost per resolution determines unit economics.'),

('Covariant', 'AI / Robotics', 'ai-startups', '201-500',
 'AI-powered warehouse robotics with human-like picking.',
 'WHY: Reinforcement learning training for robotic manipulation is notoriously GPU-wasteful. Many training runs fail and need detection.'),

('Scale AI', 'AI / Data', 'ai-startups', '1001-5000',
 'AI data labeling and model training platform for enterprises.',
 'WHY: Runs GPU workloads for government and enterprise customers — needs per-customer chargeback and compliance reporting.'),

('Weights & Biases', 'AI / MLOps', 'ml-teams', '201-500',
 'ML experiment tracking platform used by 30+ foundation model teams.',
 'WHY: Natural integration partner — their users already track experiments. AluminatAI GPU cost data enriches their dashboard.'),

('Anyscale', 'AI / Infrastructure', 'ml-teams', '201-500',
 'Ray platform for distributed ML training and inference.',
 'WHY: Distributed Ray clusters make GPU utilization monitoring complex — needs per-worker, per-job tracking across the cluster.'),

('Determined AI (HPE)', 'AI / MLOps', 'ml-teams', '201-500',
 'ML training platform with GPU scheduling and resource management.',
 'WHY: GPU scheduling efficiency is their core value prop — AluminatAI data helps prove their scheduler reduces waste.'),

('Lightning AI', 'AI / MLOps', 'ml-teams', '51-200',
 'PyTorch Lightning framework and GPU cloud platform.',
 'WHY: Their users train models on GPUs daily — cost tracking during training is a natural extension of their platform.'),

('Predibase', 'AI / Fine-Tuning', 'ai-startups', '51-200',
 'Fine-tuning platform for enterprise LLM customization.',
 'WHY: Per-customer fine-tuning cost attribution needed to price their service. Each fine-tune has different GPU requirements.'),

('Lamini', 'AI / Fine-Tuning', 'ai-startups', '11-50',
 'Enterprise LLM fine-tuning and inference platform.',
 'WHY: Needs to show customers exactly what their fine-tune costs. Transparency builds trust for enterprise sales.'),

('Baseten', 'AI / Inference', 'ai-startups', '51-200',
 'ML model inference platform with GPU autoscaling.',
 'WHY: Per-model GPU cost tracking needed for inference pricing. Idle detection catches autoscaled instances that should have scaled down.'),

('Banana.dev', 'AI / Inference', 'ai-startups', '11-50',
 'Serverless GPU inference API for ML models.',
 'WHY: Serverless means GPUs spin up/down constantly — idle detection catches instances stuck in warm-up burning money.'),

('Replicate', 'AI / Inference', 'ai-startups', '51-200',
 'Run ML models via API with pay-per-use GPU inference.',
 'WHY: Per-model, per-customer cost attribution determines if each API call is profitable at their listed prices.'),

('Cerebras Systems', 'AI / Hardware', 'ai-startups', '501-1000',
 'AI compute company with wafer-scale engine chips.',
 'WHY: Though chip-focused, their cloud service runs GPU+WSE hybrid workloads that need unified cost monitoring.'),

-- ═══════════════════════════════════════════════════════════════
-- CATEGORY 4: Autonomous Vehicles & Robotics (20)
-- ═══════════════════════════════════════════════════════════════

('Waymo', 'Autonomous Vehicles', 'ml-teams', '1001-5000',
 'Alphabets self-driving car division, largest commercial AV deployment.',
 'WHY: Massive GPU fleet for perception model training and simulation. Multiple teams (perception, prediction, planning) need chargeback.'),

('Cruise (GM)', 'Autonomous Vehicles', 'ml-teams', '1001-5000',
 'GMs autonomous vehicle division, restructuring after operational pause.',
 'WHY: Rebuilding after pause — needs cost discipline on GPU spend. Every dollar matters during restructuring.'),

('Aurora Innovation', 'Autonomous Vehicles', 'ml-teams', '1001-5000',
 'Self-driving technology for trucks and ride-hailing.',
 'WHY: Long training runs for highway perception models tie up GPUs for days. Waste detection catches jobs that stall silently.'),

('Nuro', 'Autonomous Vehicles', 'ml-teams', '501-1000',
 'Autonomous delivery vehicles for last-mile logistics.',
 'WHY: Smaller GPU fleet but costs still significant on startup budget. Budget alerts prevent overspend.'),

('Zoox (Amazon)', 'Autonomous Vehicles', 'ml-teams', '1001-5000',
 'Amazons robotaxi division building purpose-built autonomous vehicles.',
 'WHY: Multiple research teams (perception, simulation, planning) sharing GPU clusters need fair chargeback between groups.'),

('Motional', 'Autonomous Vehicles', 'ml-teams', '501-1000',
 'Hyundai/Aptiv joint venture for autonomous driving.',
 'WHY: Joint venture structure means transparent GPU cost attribution between Hyundai and Aptiv is contractually important.'),

('Mobileye', 'Autonomous Vehicles', 'ml-teams', '5001-10000',
 'Intels autonomous driving technology division.',
 'WHY: EyeQ chip training pipeline runs 24/7 on GPU clusters. Large organization needs per-team utilization reporting.'),

('Pony.ai', 'Autonomous Vehicles', 'ml-teams', '501-1000',
 'Chinese autonomous driving company expanding to US operations.',
 'WHY: Multi-region GPU infrastructure (China + US) needs centralized monitoring across geographies.'),

('Kodiak Robotics', 'Autonomous Vehicles', 'ml-teams', '201-500',
 'Autonomous trucking for long-haul freight and defense.',
 'WHY: Defense contracts (US Army) require detailed cost reporting per project. GPU cost attribution is compliance-required.'),

('Gatik', 'Autonomous Vehicles', 'ml-teams', '201-500',
 'Middle-mile autonomous delivery for retail (Walmart, Loblaw).',
 'WHY: Startup budget means GPU training costs must be tracked carefully. Cant afford runaway experiments.'),

('Waabi', 'Autonomous Vehicles', 'ml-teams', '201-500',
 'Generative AI approach to self-driving (simulation-first).',
 'WHY: Simulation-heavy approach is extremely GPU-intensive — their entire methodology depends on efficient GPU utilization.'),

('Ghost Autonomy', 'Autonomous Vehicles', 'ml-teams', '51-200',
 'Highway autonomy technology for consumer vehicles.',
 'WHY: Small team iterating fast on perception models — budget alerts catch expensive training mistakes before they drain runway.'),

('Boston Dynamics', 'Robotics', 'ml-teams', '1001-5000',
 'Advanced robotics (Atlas, Spot, Stretch) with AI-powered movement.',
 'WHY: Reinforcement learning for robot locomotion and manipulation consumes significant GPU compute over long training runs.'),

('Figure AI', 'Robotics', 'ml-teams', '201-500',
 'Humanoid robot startup training whole-body control with AI.',
 'WHY: Training humanoid robots requires massive RL compute. GPU costs are their second-largest expense after hardware.'),

('Agility Robotics', 'Robotics', 'ml-teams', '201-500',
 'Digit humanoid robot for warehouse logistics.',
 'WHY: RL training for bipedal walking and package manipulation. Long training runs that often fail need waste detection.'),

('1X Technologies', 'Robotics', 'ml-teams', '201-500',
 'NEO humanoid robot backed by OpenAI.',
 'WHY: OpenAI partnership means access to large GPU clusters — needs monitoring to use that expensive compute wisely.'),

('Skydio', 'Robotics / Drones', 'ml-teams', '501-1000',
 'Autonomous drones with AI-powered visual navigation.',
 'WHY: Visual navigation model training on GPU clusters. Defense/enterprise customers require cost documentation.'),

('Shield AI', 'Robotics / Defense', 'ml-teams', '501-1000',
 'AI-powered autonomous defense systems and drones.',
 'WHY: Government contracts require detailed per-project GPU cost reporting. Carbon tracking aligns with DoD sustainability goals.'),

('Outrider', 'Robotics', 'ml-teams', '201-500',
 'Autonomous yard trucks for distribution centers.',
 'WHY: Training perception models for industrial environments. Startup budget means every GPU-hour counts.'),

('Apptronik', 'Robotics', 'ml-teams', '201-500',
 'Apollo humanoid robot for manufacturing and logistics.',
 'WHY: New entrant training humanoid control models — needs cost visibility as GPU spend ramps from zero.'),

-- ═══════════════════════════════════════════════════════════════
-- CATEGORY 5: Biotech & Pharma (20)
-- ═══════════════════════════════════════════════════════════════

('Recursion Pharmaceuticals', 'Biotech', 'ml-teams', '1001-5000',
 'AI-driven drug discovery with $850M for compute post-Exscientia merger.',
 'WHY: One of biotechs largest in-house GPU deployments. BioHive-2 supercomputer needs utilization tracking across dozens of research teams.'),

('Schrodinger', 'Biotech / Simulation', 'ml-teams', '1001-5000',
 'Physics-based molecular simulation platform for drug design.',
 'WHY: GPU clusters run molecular dynamics simulations 24/7. Long-running jobs that stall waste thousands in GPU-hours.'),

('Isomorphic Labs', 'Biotech', 'ml-teams', '201-500',
 'DeepMind spinoff applying AI to drug discovery (AlphaFold).',
 'WHY: AlphaFold-scale training is massively GPU-intensive. Per-experiment cost tracking for investors and board reporting.'),

('Relay Therapeutics', 'Biotech', 'ml-teams', '501-1000',
 'AI-driven drug design using molecular dynamics on GPUs.',
 'WHY: GPU molecular dynamics simulations for protein targets run for weeks. Waste detection catches simulations that diverge.'),

('Insilico Medicine', 'Biotech', 'ml-teams', '201-500',
 'AI drug discovery with drugs in clinical trials.',
 'WHY: Generative chemistry models trained on GPUs. Clinical-stage company needs to justify compute spend to investors.'),

('Absci', 'Biotech', 'ml-teams', '201-500',
 'Generative AI for antibody and drug design.',
 'WHY: Protein folding and antibody design models require large GPU compute. Per-project tracking for pharma partnership billing.'),

('Generate Biomedicines', 'Biotech', 'ml-teams', '201-500',
 'Generative biology platform for protein therapeutics.',
 'WHY: Generative protein models are among the most GPU-intensive biotech workloads. Cost visibility drives research prioritization.'),

('BioMap', 'Biotech', 'ml-teams', '201-500',
 'Cross-modal life sciences AI (xTrimo model covering DNA, RNA, protein).',
 'WHY: Training foundation models across 7 biological modalities requires massive GPU clusters with per-modality cost tracking.'),

('Eli Lilly', 'Pharma', 'ml-teams', '10001+',
 'Pharma giant with 1,016 Blackwell Ultra GPU supercomputer — industrys largest.',
 'WHY: Pharmas most powerful supercomputer needs monitoring. Multiple drug programs sharing the cluster need chargeback.'),

('Amgen (deCODE Genetics)', 'Pharma', 'ml-teams', '10001+',
 'Pharma with DGX SuperPOD (248 H100s) at deCODE in Iceland.',
 'WHY: Freyja supercomputer used by multiple genomics teams — fair GPU allocation and cost tracking across programs.'),

('Novo Nordisk', 'Pharma', 'ml-teams', '10001+',
 'Pharma using Gefion supercomputer (1,528 H100s) for drug discovery.',
 'WHY: Shared sovereign supercomputer — needs per-team chargeback. Danish sustainability reporting requires carbon tracking.'),

('AstraZeneca', 'Pharma', 'ml-teams', '10001+',
 'Global pharma with growing AI drug discovery program.',
 'WHY: GPU clusters for ADMET prediction and molecular generation shared across therapeutic areas need cost attribution.'),

('Roche / Genentech', 'Pharma', 'ml-teams', '10001+',
 'Pharma with large internal ML teams for drug discovery and diagnostics.',
 'WHY: Multiple divisions (pharma, diagnostics) sharing GPU resources. Swiss HQ means EU sustainability compliance.'),

('Pfizer', 'Pharma', 'ml-teams', '10001+',
 'Global pharma with growing AI program for molecular modeling.',
 'WHY: Expanding GPU usage for clinical trial optimization and molecular modeling. Needs cost visibility as AI spend grows.'),

('BenevolentAI', 'Biotech', 'ml-teams', '201-500',
 'AI drug discovery platform with drugs in clinical trials.',
 'WHY: GPU training costs are core operating expense. London-listed company needs to report compute spend to shareholders.'),

('Tempus AI', 'Healthcare / AI', 'ml-teams', '1001-5000',
 'Precision medicine using genomics and ML for cancer treatment.',
 'WHY: GPU-intensive genomic analysis and ML training. Recently IPOd — public company needs detailed compute cost reporting.'),

('Owkin', 'Biotech', 'ml-teams', '201-500',
 'Federated learning platform for pharma R&D.',
 'WHY: Federated GPU training across hospital sites — needs per-site and per-model cost tracking in distributed environments.'),

('PathAI', 'Healthcare / AI', 'ml-teams', '201-500',
 'AI pathology for cancer diagnosis from tissue images.',
 'WHY: Training on massive histopathology image datasets (terabytes of slides) is GPU-intensive. Cost-per-model tracking needed.'),

('Chai Discovery', 'Biotech', 'ml-teams', '51-200',
 'AI platform for molecular structure prediction and drug design.',
 'WHY: GPU-intensive protein structure prediction. Early-stage startup needs to monitor GPU spend carefully.'),

('Basecamp Research', 'Biotech', 'ml-teams', '51-200',
 'Biodiversity-powered drug discovery using protein AI models.',
 'WHY: Training protein models on biological data. Small team needs turnkey GPU monitoring without building it themselves.'),

-- ═══════════════════════════════════════════════════════════════
-- CATEGORY 6: Quantitative Finance (15)
-- ═══════════════════════════════════════════════════════════════

('Citadel', 'Quantitative Finance', 'ml-teams', '5001-10000',
 'Worlds largest hedge fund with massive private GPU infrastructure.',
 'WHY: Private GPU data center for alpha generation. Every wasted GPU cycle is lost trading signal. Utilization optimization = money.'),

('Two Sigma', 'Quantitative Finance', 'ml-teams', '1001-5000',
 'Technology-driven hedge fund using ML for investment decisions.',
 'WHY: GPU clusters for backtesting and portfolio optimization. Multiple quant teams sharing GPUs need fair allocation tracking.'),

('DE Shaw', 'Quantitative Finance', 'ml-teams', '1001-5000',
 'Pioneering quant fund. DE Shaw Research also does molecular simulation.',
 'WHY: Dual GPU usage — trading AND molecular simulation (DESRES). Two very different workload profiles need separate cost tracking.'),

('Jane Street', 'Quantitative Finance', 'ml-teams', '1001-5000',
 'Quantitative trading firm using GPU-accelerated pricing models.',
 'WHY: GPU-accelerated derivatives pricing and strategy backtesting. Latency-sensitive — idle GPUs mean missed trades.'),

('Renaissance Technologies', 'Quantitative Finance', 'ml-teams', '201-500',
 'Legendary Medallion fund, heavily ML-driven trading.',
 'WHY: Most successful quant fund ever — any GPU optimization directly improves already-legendary returns.'),

('Point72', 'Quantitative Finance', 'ml-teams', '1001-5000',
 'Steve Cohens multi-strategy hedge fund with growing ML team.',
 'WHY: Rapidly growing ML division means GPU spend is ramping. Needs cost visibility before it becomes unmanageable.'),

('Man Group (AHL)', 'Quantitative Finance', 'ml-teams', '1001-5000',
 'Worlds largest publicly traded hedge fund with ML research division.',
 'WHY: Public company — GPU compute costs show up on financial statements. Needs to justify spend to shareholders.'),

('Bridgewater Associates', 'Quantitative Finance', 'ml-teams', '1001-5000',
 'Worlds largest hedge fund by AUM, growing AI capabilities.',
 'WHY: Building out AI team from scratch — needs monitoring infrastructure as GPU fleet grows from zero.'),

('Jump Trading', 'Quantitative Finance', 'ml-teams', '501-1000',
 'High-frequency trading firm with GPU-accelerated strategies.',
 'WHY: HFT means GPUs run 24/7 for signal generation. Waste detection catches jobs that stall during market hours.'),

('Tower Research Capital', 'Quantitative Finance', 'ml-teams', '501-1000',
 'Quantitative trading with low-latency GPU infrastructure.',
 'WHY: Low-latency GPU inference for trading signals — needs to track which models are worth the GPU cost vs. underperforming.'),

('Hudson River Trading', 'Quantitative Finance', 'ml-teams', '501-1000',
 'Quantitative trading firm using GPUs for model training.',
 'WHY: GPU clusters for continuous model retraining. Cost attribution per strategy helps determine which ones justify compute.'),

('Millennium Management', 'Quantitative Finance', 'ml-teams', '1001-5000',
 'Multi-strategy hedge fund with 300+ independent trading pods.',
 'WHY: 300+ pods sharing GPU infrastructure — chargeback per pod is essential for their operating model.'),

('WorldQuant', 'Quantitative Finance', 'ml-teams', '1001-5000',
 'Quantitative asset management with globally distributed research.',
 'WHY: Distributed research teams across 25+ offices sharing GPU resources — per-team cost attribution across geographies.'),

('Qube Research & Technologies', 'Quantitative Finance', 'ml-teams', '501-1000',
 'Systematic trading firm using GPU-intensive research.',
 'WHY: GPU-intensive factor model research. European HQ means ESG reporting on compute energy usage.'),

('Squarepoint Capital', 'Quantitative Finance', 'ml-teams', '501-1000',
 'Systematic trading firm with GPU clusters for model training.',
 'WHY: Competing for alpha on GPU speed — utilization monitoring ensures no GPU cycles wasted on stale strategies.'),

-- ═══════════════════════════════════════════════════════════════
-- CATEGORY 7: Enterprise AI Teams (25)
-- ═══════════════════════════════════════════════════════════════

('Tesla', 'Automotive / AI', 'ml-teams', '10001+',
 'EV maker with Dojo supercomputer and NVIDIA clusters for FSD training.',
 'WHY: Massive GPU fleet for Full Self-Driving training. Autopilot vs. Optimus vs. Dojo teams need GPU chargeback.'),

('Microsoft', 'Big Tech', 'ml-teams', '10001+',
 'Azure AI, GitHub Copilot, and internal ML teams.',
 'WHY: Internal GPU clusters shared across Azure, GitHub, Office, and Bing teams. Cross-division chargeback is a nightmare without tooling.'),

('Netflix', 'Entertainment', 'ml-teams', '10001+',
 'Streaming giant with ML-driven recommendations and content AI.',
 'WHY: Recommendation systems, content quality, and encoding — multiple ML teams sharing GPU clusters need cost visibility.'),

('Spotify', 'Entertainment', 'ml-teams', '5001-10000',
 'Music streaming with ML-heavy recommendation engine.',
 'WHY: Podcast transcription, music recommendations, and search — GPU training across multiple ML teams needs chargeback.'),

('Uber', 'Transportation', 'ml-teams', '10001+',
 'Ride-sharing with ML platform powering pricing, routing, and ETA.',
 'WHY: ML platform team runs GPU workloads for pricing, fraud, maps, and Eats — per-team cost attribution required.'),

('Airbnb', 'Travel', 'ml-teams', '5001-10000',
 'Travel platform with ML-driven search ranking and pricing.',
 'WHY: Search ranking, dynamic pricing, and trust/safety models all trained on GPUs. Growing AI spend needs tracking.'),

('Snap', 'Social Media', 'ml-teams', '5001-10000',
 'Snapchat with AR filters and ML-powered content recommendations.',
 'WHY: Real-time ML inference for AR lenses and Spotlight. GPU costs scale with daily active users.'),

('Pinterest', 'Social Media', 'ml-teams', '5001-10000',
 'Visual discovery platform with GPU-intensive image understanding.',
 'WHY: Visual search and recommendation models trained on billions of images. GPU training is core infrastructure cost.'),

('Twitter / X', 'Social Media', 'ml-teams', '1001-5000',
 'Social platform with Grok AI and recommendation algorithms.',
 'WHY: Grok training plus recommendation algorithm GPUs — multiple AI products sharing fleet need cost attribution.'),

('Shopify', 'E-commerce', 'ml-teams', '10001+',
 'E-commerce platform adding AI features (Sidekick, Magic).',
 'WHY: Growing GPU footprint for AI shopping assistant and product descriptions. New AI spend needs cost governance.'),

('Salesforce', 'Enterprise Software', 'ml-teams', '10001+',
 'CRM giant with Einstein AI and CodeGen models.',
 'WHY: Enterprise AI features trained on GPU clusters. Multiple product teams (Sales, Service, Marketing) need GPU chargeback.'),

('Adobe', 'Creative Software', 'ml-teams', '10001+',
 'Creative suite with Firefly AI image generation.',
 'WHY: Massive GPU inference fleet for Firefly across Creative Cloud. Cost-per-generation determines AI feature profitability.'),

('Intuit', 'Finance Software', 'ml-teams', '10001+',
 'TurboTax and QuickBooks with growing AI capabilities.',
 'WHY: Financial AI models for tax assistance and bookkeeping trained on GPUs. Regulated industry needs cost audit trails.'),

('ServiceNow', 'Enterprise Software', 'ml-teams', '10001+',
 'Enterprise workflow platform with Now Assist AI.',
 'WHY: Growing GPU usage for LLM fine-tuning per customer. Per-tenant GPU cost attribution for enterprise pricing.'),

('Palantir', 'Data Analytics', 'ml-teams', '5001-10000',
 'AIP platform for government and enterprise AI applications.',
 'WHY: Government contracts require detailed GPU cost reporting. Multiple classified workloads need separate cost tracking.'),

('Snowflake', 'Data Platform', 'ml-teams', '5001-10000',
 'Cloud data platform with Cortex AI for ML features.',
 'WHY: GPU-backed ML features (Cortex) need per-customer cost tracking to price their compute-heavy AI tier.'),

('Block (Square)', 'Fintech', 'ml-teams', '10001+',
 'Payment platform with GPU-trained fraud detection models.',
 'WHY: Real-time fraud scoring on GPUs. Cost-per-transaction for ML determines if AI fraud detection is economically viable.'),

('Stripe', 'Fintech', 'ml-teams', '5001-10000',
 'Payment infrastructure with GPU-trained Radar fraud detection.',
 'WHY: Radar processes billions of transactions — GPU inference costs at that scale need careful monitoring.'),

('DoorDash', 'Delivery', 'ml-teams', '5001-10000',
 'Food delivery with ML for logistics, recommendations, and pricing.',
 'WHY: GPU training for delivery time prediction, restaurant ranking, and dynamic pricing. Multiple ML teams sharing GPUs.'),

('Walmart', 'Retail', 'ml-teams', '10001+',
 'Largest retailer with AI for supply chain, demand forecasting, and CV.',
 'WHY: Massive scale — AI for supply chain, inventory, and computer vision across 10K+ stores needs GPU cost governance.'),

('Target', 'Retail', 'ml-teams', '10001+',
 'Retail with ML-driven inventory management and personalization.',
 'WHY: Growing GPU infrastructure for demand forecasting and personalization. Needs cost tracking as AI investment scales.'),

('JPMorgan Chase', 'Banking', 'ml-teams', '10001+',
 'Largest US bank with massive AI research division (200+ ML researchers).',
 'WHY: GPU clusters for NLP, fraud detection, and risk modeling. Regulated industry requires detailed compute cost auditing.'),

('Goldman Sachs', 'Banking', 'ml-teams', '10001+',
 'Investment bank with internal ML platform for trading and risk.',
 'WHY: GPU clusters for trading signals, risk models, and NLP. Multiple divisions sharing GPUs need transparent chargeback.'),

('Morgan Stanley', 'Banking', 'ml-teams', '10001+',
 'Investment bank deploying AI assistant and research tools firm-wide.',
 'WHY: Growing GPU usage for internal AI tools. Banking regulators expect detailed technology cost reporting.'),

('Visa', 'Payments', 'ml-teams', '10001+',
 'Payment network using GPU-trained models for fraud detection.',
 'WHY: Processes 65K+ transactions per second — GPU inference for real-time fraud scoring at scale needs cost monitoring.'),

-- ═══════════════════════════════════════════════════════════════
-- CATEGORY 8: Research Labs & Universities (15)
-- ═══════════════════════════════════════════════════════════════

('MIT CSAIL', 'Academic Research', 'research-labs', '1001-5000',
 'MITs Computer Science and AI Lab — premier AI research institution.',
 'WHY: Shared GPU cluster with 100+ researchers and no cost visibility. Fair usage tracking prevents tragedy of the commons.'),

('Stanford HAI', 'Academic Research', 'research-labs', '501-1000',
 'Stanford Human-Centered AI Institute.',
 'WHY: GPU cluster shared by 100+ researchers across departments. Grant funding requires compute cost reporting per project.'),

('UC Berkeley BAIR', 'Academic Research', 'research-labs', '501-1000',
 'Berkeley AI Research Lab — top ML and robotics research.',
 'WHY: Per-project GPU cost reporting needed for NSF/DARPA grant compliance. Multiple PIs sharing limited GPU resources.'),

('CMU ML Department', 'Academic Research', 'research-labs', '1001-5000',
 'Carnegie Mellons Machine Learning Department.',
 'WHY: Shared GPUs with no chargeback means some researchers hog resources. Fair allocation tracking solves department politics.'),

('UT Austin TACC', 'Academic Research', 'research-labs', '201-500',
 'Texas Advanced Computing Center with 5,000+ GPUs — largest academic GPU deployment.',
 'WHY: 5,000+ GPUs shared across entire university system. Utilization monitoring at this scale requires dedicated tooling.'),

('University of Washington', 'Academic Research', 'research-labs', '1001-5000',
 'Allen Institute partnership for NLP research.',
 'WHY: Shared GPU infrastructure between UW and Allen Institute. Cross-org cost attribution for joint research projects.'),

('ETH Zurich', 'Academic Research', 'research-labs', '1001-5000',
 'Top European research university with growing AI program.',
 'WHY: Swiss sustainability requirements plus shared GPU cluster across departments. European compliance needs carbon tracking.'),

('MILA Montreal', 'Academic Research', 'research-labs', '501-1000',
 'Yoshua Bengios Quebec AI Institute — one of the worlds top ML labs.',
 'WHY: Large GPU cluster shared across many research groups. Canadian government funding requires compute cost documentation.'),

('Princeton AI Lab', 'Academic Research', 'research-labs', '201-500',
 'Recently invested in 300-GPU cluster for academic AI research.',
 'WHY: Brand new 300-GPU cluster — perfect timing to deploy monitoring from day one rather than retrofit later.'),

('Georgia Tech', 'Academic Research', 'research-labs', '1001-5000',
 'ML and robotics research with shared GPU clusters.',
 'WHY: Shared cluster across CS, ECE, and ISYE departments. Cross-department chargeback needed for fair resource allocation.'),

('Stony Brook NVwulf', 'Academic Research', 'research-labs', '1001-5000',
 'New HPC cluster with 24x NVIDIA H200 GPUs for AI research.',
 'WHY: Fresh H200 deployment — new infrastructure is the ideal time to set up monitoring. 80 petaFLOPs needs tracking.'),

('Allen Institute for AI (AI2)', 'Research', 'research-labs', '201-500',
 'Non-profit AI research lab (OLMo, Semantic Scholar).',
 'WHY: Non-profit means every GPU dollar must be justified to donors. Waste detection maximizes research output per dollar.'),

('KAIST', 'Academic Research', 'research-labs', '1001-5000',
 'South Koreas top science university with growing AI GPU infrastructure.',
 'WHY: Expanding GPU clusters for AI research. Korean government funding for AI requires utilization reporting.'),

('Tsinghua University', 'Academic Research', 'research-labs', '10001+',
 'Chinas top CS university with massive GPU clusters for NLP and vision.',
 'WHY: Largest academic AI research program in China. Thousands of researchers sharing GPU resources need fair allocation.'),

('National Labs (ORNL/ANL/LLNL)', 'Government Research', 'research-labs', '10001+',
 'US DOE national laboratories with GPU-heavy supercomputers (Frontier, Aurora).',
 'WHY: Frontier (37K+ GPUs) and Aurora — government supercomputers need per-project cost attribution and energy reporting.'),

-- ═══════════════════════════════════════════════════════════════
-- CATEGORY 9: VFX, Gaming & Rendering (15)
-- ═══════════════════════════════════════════════════════════════

('Industrial Light & Magic', 'VFX', 'cloud-users', '1001-5000',
 'Lucasfilms VFX studio (Star Wars, Marvel). GPU render farms.',
 'WHY: GPU render farm needs cost-per-shot and cost-per-show tracking. Failed renders still consuming GPUs waste money.'),

('Weta FX', 'VFX', 'cloud-users', '1001-5000',
 'Avatar and Lord of the Rings VFX. Massive GPU rendering pipeline.',
 'WHY: Largest VFX renders in film history. Cost attribution per project (Avatar vs. other films) needed for client billing.'),

('DNEG', 'VFX', 'cloud-users', '5001-10000',
 'Oscar-winning VFX studio with GPU render farms across multiple offices.',
 'WHY: Multi-site GPU render farms (London, Mumbai, Montreal, LA). Per-office and per-project cost tracking needed.'),

('Framestore', 'VFX', 'cloud-users', '1001-5000',
 'VFX studio with GPU rendering and ML-enhanced workflows.',
 'WHY: Transitioning to GPU rendering + ML denoising. Hybrid workloads need unified cost monitoring.'),

('Epic Games', 'Gaming', 'cloud-users', '1001-5000',
 'Unreal Engine developer with internal GPU farm.',
 'WHY: Internal GPU clusters for Unreal Engine development and MetaHuman neural network training. R&D cost tracking.'),

('Unity Technologies', 'Gaming', 'cloud-users', '5001-10000',
 'Game engine with ML-powered graphics features.',
 'WHY: GPU clusters for training neural rendering and ML graphics features. Cost visibility for AI R&D investment.'),

('Electronic Arts', 'Gaming', 'ml-teams', '10001+',
 'Major game publisher using GPU ML for player modeling and content.',
 'WHY: GPU clusters shared across multiple game studios. Per-studio chargeback for ML infrastructure costs.'),

('Ubisoft', 'Gaming', 'ml-teams', '10001+',
 'Game publisher with GPU ML for NPC AI and procedural generation.',
 'WHY: ML for NPC behavior, procedural worlds, and anti-cheat. Multiple studios sharing GPU infrastructure need tracking.'),

('Riot Games', 'Gaming', 'ml-teams', '5001-10000',
 'League of Legends and Valorant developer using ML models.',
 'WHY: ML for matchmaking, anti-cheat (Vanguard), and player behavior. GPU costs for ML across game titles need attribution.'),

('Roblox', 'Gaming', 'ml-teams', '5001-10000',
 'User-generated content platform with GPU inference for safety.',
 'WHY: GPU inference for content moderation at scale — billions of user-generated items need AI screening. Cost-per-moderation matters.'),

('GarageFarm.net', 'Rendering', 'cloud-users', '51-200',
 'Cloud render farm service for 3D artists and studios.',
 'WHY: Per-customer GPU cost attribution for render billing. Idle detection between render jobs optimizes fleet utilization.'),

('RebusFarm', 'Rendering', 'cloud-users', '51-200',
 'Cloud rendering service with GPU-accelerated pipelines.',
 'WHY: GPU utilization monitoring determines pricing and profitability. Waste between jobs directly impacts margins.'),

('OTOY (Render Network)', 'Rendering', 'cloud-users', '201-500',
 'Decentralized GPU rendering network for VFX and metaverse.',
 'WHY: Decentralized network of GPU nodes — needs per-node monitoring across thousands of distributed render machines.'),

('Wonder Dynamics', 'VFX / AI', 'ai-startups', '51-200',
 'AI VFX tool for real-time visual effects replacement.',
 'WHY: GPU inference for real-time VFX processing. AI-powered pipeline needs per-shot cost tracking.'),

('Pixar (Disney)', 'Animation', 'cloud-users', '1001-5000',
 'Animation studio transitioning to GPU-accelerated rendering.',
 'WHY: Moving from CPU to GPU rendering. New GPU infrastructure needs monitoring as they build out the fleet.'),

-- ═══════════════════════════════════════════════════════════════
-- CATEGORY 10: Climate, Energy & ESG (10)
-- ═══════════════════════════════════════════════════════════════

('ClimateAI', 'Climate Tech', 'custom', '51-200',
 'AI for climate risk assessment and supply chain resilience.',
 'WHY: Practices what they preach — needs to demonstrate low-carbon AI operations. Carbon tracking validates their mission.'),

('Pachama', 'Climate Tech', 'custom', '51-200',
 'Forest carbon credits verified by ML on satellite imagery.',
 'WHY: GPU training for satellite image analysis. Must show their own AI operations are carbon-efficient to maintain credibility.'),

('Watershed', 'Climate Tech', 'custom', '201-500',
 'Enterprise carbon accounting platform used by major corporations.',
 'WHY: Could integrate AluminatAI data to help their customers report GPU compute emissions. Natural partnership opportunity.'),

('Persefoni', 'Climate Tech', 'custom', '201-500',
 'Carbon management and ESG reporting platform.',
 'WHY: Needs GPU carbon tracking for their own ML training. Also a potential integration — their platform could ingest our data.'),

('WattTime', 'Energy / Climate', 'custom', '11-50',
 'Grid carbon intensity API providing real-time marginal emissions data.',
 'WHY: Natural data partner — their API already feeds carbon intensity data. Integration could make GPU scheduling carbon-aware.'),

('CarbonChain', 'Climate Tech', 'custom', '51-200',
 'Supply chain carbon tracking platform.',
 'WHY: GPU-trained ML models for emissions estimation. Their own compute carbon footprint should be tracked and reported.'),

('Normative', 'Climate Tech', 'custom', '51-200',
 'Carbon accounting for EU CSRD compliance.',
 'WHY: EU CSRD reporting requires tracking energy usage of compute workloads. AluminatAI provides the GPU-specific data they need.'),

('Electricity Maps', 'Energy / Climate', 'custom', '51-200',
 'Real-time global carbon intensity data and API.',
 'WHY: Potential integration partner — their grid carbon data combined with our GPU energy data enables carbon-aware compute scheduling.'),

('BCG Gamma', 'Consulting', 'custom', '1001-5000',
 'BCGs AI consulting division advising Fortune 500 on AI + sustainability.',
 'WHY: Advises F500 companies on AI costs and sustainability. Could recommend AluminatAI to clients running GPU infrastructure.'),

('McKinsey QuantumBlack', 'Consulting', 'custom', '1001-5000',
 'McKinseys AI division building GPU-heavy solutions for clients.',
 'WHY: Builds custom AI solutions on GPU clusters for engagements. Needs per-client GPU cost tracking for project billing and carbon reporting.')

ON CONFLICT DO NOTHING;

COMMIT;
