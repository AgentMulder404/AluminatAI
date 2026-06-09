# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""CLI handlers for `nemulai model-intel`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nemulai model-intel",
        description="Model intelligence pipeline — discover and profile AI models.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    scan_p = sub.add_parser("scan", help="Scan for new model releases")
    scan_p.add_argument("--limit", type=int, default=20)
    scan_p.add_argument("--min-downloads", type=int, default=1000)
    scan_p.add_argument("--min-confidence", type=float, default=0.5)
    scan_p.add_argument("--json", action="store_true", dest="json_output")

    list_p = sub.add_parser("list", help="List discovered models")
    list_p.add_argument(
        "--status",
        choices=["detected", "profiled", "estimated", "active", "all"],
        default="all",
    )
    list_p.add_argument("--json", action="store_true", dest="json_output")

    pair_p = sub.add_parser("pair", help="Profile and rank GPUs for a model")
    pair_p.add_argument("model", help="HuggingFace model ID or local tag")
    pair_p.add_argument("--top", type=int, default=10)
    pair_p.add_argument("--json", action="store_true", dest="json_output")

    profile_p = sub.add_parser("profile", help="Show profiling details for a model")
    profile_p.add_argument("model", help="HuggingFace model ID")
    profile_p.add_argument("--json", action="store_true", dest="json_output")

    sub.add_parser("warm-start", help="Warm-start bandit with discovered model profiles")

    quant_p = sub.add_parser("quantize", help="Analyze quantization variants for a model")
    quant_p.add_argument("model", help="HuggingFace model ID or local tag")
    quant_p.add_argument("--gpu", default=None, help="Show recommendation for specific GPU")
    quant_p.add_argument("--json", action="store_true", dest="json_output")

    prices_p = sub.add_parser("prices", help="Show current GPU pricing")
    prices_p.add_argument("--update-from", default=None, help="Ingest pricing from JSON file")
    prices_p.add_argument("--json", action="store_true", dest="json_output")

    bv_p = sub.add_parser("best-value", help="Find best $/TFLOP GPU for a model")
    bv_p.add_argument("model", help="HuggingFace model ID or local tag")
    bv_p.add_argument("--budget", type=float, default=None, help="Max $/hr budget")
    bv_p.add_argument("--top", type=int, default=10)
    bv_p.add_argument("--json", action="store_true", dest="json_output")

    return parser


def run_model_intel(args: argparse.Namespace) -> int:
    from config import DATA_DIR, SUPABASE_URL, SUPABASE_SERVICE_KEY
    from intelligence.pipeline import IntelligencePipeline

    data_dir = Path(DATA_DIR) if DATA_DIR else Path.home() / ".nemulai"
    pipeline = IntelligencePipeline(
        data_dir=data_dir,
        supabase_url=SUPABASE_URL if hasattr(sys.modules.get("config", object), "SUPABASE_URL") else None,
        supabase_key=SUPABASE_SERVICE_KEY if hasattr(sys.modules.get("config", object), "SUPABASE_SERVICE_KEY") else None,
    )

    if args.action == "scan":
        return _cmd_scan(pipeline, args)
    elif args.action == "list":
        return _cmd_list(pipeline, args)
    elif args.action == "pair":
        return _cmd_pair(pipeline, args)
    elif args.action == "profile":
        return _cmd_profile(pipeline, args)
    elif args.action == "warm-start":
        return _cmd_warm_start(pipeline, args)
    elif args.action == "quantize":
        return _cmd_quantize(pipeline, args)
    elif args.action == "prices":
        return _cmd_prices(data_dir, args)
    elif args.action == "best-value":
        return _cmd_best_value(pipeline, data_dir, args)

    return 1


def _cmd_scan(pipeline, args) -> int:
    console = Console()
    console.print("[bold]Scanning HuggingFace for new models...[/bold]\n")

    result = pipeline.run(
        limit=args.limit,
        min_downloads=args.min_downloads,
        min_confidence=args.min_confidence,
    )

    if args.json_output:
        print(json.dumps({
            "detected": result.detected,
            "profiled": result.profiled,
            "estimated": result.estimated,
            "registered": result.registered,
            "duration_s": result.duration_s,
            "errors": result.errors,
            "models": [e.to_dict() for e in result.entries],
        }, indent=2))
        return 0

    if not result.entries:
        console.print(f"No new models found ({result.detected} detected, filtered by confidence/downloads)")
        return 0

    table = Table(title=f"Discovered {len(result.entries)} New Models ({result.duration_s}s)")
    table.add_column("Tag", style="cyan")
    table.add_column("Family", style="green")
    table.add_column("Params")
    table.add_column("Intensity", justify="right")
    table.add_column("Precision")
    table.add_column("Best GPU", style="yellow")
    table.add_column("Confidence", justify="right")

    for entry in result.entries:
        params = f"{entry.parameter_count / 1e9:.1f}B" if entry.parameter_count else "?"
        best = entry.gpu_rankings[0]["gpu_name"] if entry.gpu_rankings else "?"
        table.add_row(
            entry.tag,
            entry.family,
            params,
            f"{entry.profile.math_intensity:.0f}",
            entry.profile.precision,
            best,
            f"{entry.confidence:.0%}",
        )

    console.print(table)

    if result.errors:
        console.print(f"\n[yellow]{len(result.errors)} errors:[/yellow]")
        for err in result.errors:
            console.print(f"  - {err}")

    return 0


def _cmd_list(pipeline, args) -> int:
    console = Console()
    entries = pipeline.registry.list_all(status=args.status)

    if args.json_output:
        print(json.dumps([e.to_dict() for e in entries], indent=2))
        return 0

    if not entries:
        console.print("No models in registry.")
        return 0

    table = Table(title=f"Model Registry ({len(entries)} models)")
    table.add_column("Tag", style="cyan")
    table.add_column("Family", style="green")
    table.add_column("Intensity", justify="right")
    table.add_column("Precision")
    table.add_column("Status", style="magenta")
    table.add_column("Best GPU", style="yellow")
    table.add_column("Downloads", justify="right")

    for entry in entries:
        best = entry.gpu_rankings[0]["gpu_name"] if entry.gpu_rankings else "?"
        downloads = f"{entry.downloads_30d:,}" if entry.downloads_30d else "?"
        table.add_row(
            entry.tag,
            entry.family,
            f"{entry.profile.math_intensity:.0f}",
            entry.profile.precision,
            entry.status,
            best,
            downloads,
        )

    console.print(table)
    return 0


def _cmd_pair(pipeline, args) -> int:
    console = Console()
    model_id = args.model

    console.print(f"[bold]Profiling and ranking GPUs for {model_id}...[/bold]\n")

    entry = pipeline.run_single(model_id)
    if not entry:
        console.print(f"[red]Could not profile model: {model_id}[/red]")
        return 1

    if args.json_output:
        print(json.dumps(entry.to_dict(), indent=2))
        return 0

    console.print(f"[bold cyan]{entry.tag}[/bold cyan] ({entry.family})")
    console.print(f"  Intensity: {entry.profile.math_intensity:.1f} FLOP/byte")
    console.print(f"  Precision: {entry.profile.precision}")
    console.print(f"  {'Memory' if entry.profile.is_memory_bound else 'Compute'}-bound")
    console.print(f"  Utilization range: {entry.profile.typical_util_min}-{entry.profile.typical_util_max}%")
    console.print()

    table = Table(title=f"Top {min(args.top, len(entry.gpu_rankings))} GPUs for {entry.tag}")
    table.add_column("#", justify="right", style="dim")
    table.add_column("GPU", style="cyan")
    table.add_column("Family")
    table.add_column("Score", justify="right", style="green")
    table.add_column("J/TFLOP", justify="right")
    table.add_column("Eff. TFLOPS", justify="right")
    table.add_column("$/hr", justify="right", style="yellow")

    for i, r in enumerate(entry.gpu_rankings[:args.top], 1):
        cost = f"${r['cost_per_hr']:.2f}" if r.get("cost_per_hr") else "?"
        table.add_row(
            str(i),
            r["gpu_name"],
            r.get("family", ""),
            f"{r['score']:.1f}",
            f"{r['joules_per_tflop']:.2f}",
            f"{r.get('effective_tflops', 0):.1f}",
            cost,
        )

    console.print(table)
    return 0


def _cmd_profile(pipeline, args) -> int:
    console = Console()
    model_id = args.model

    from intelligence.detector import ModelDetector
    from intelligence.profiler import ModelProfiler

    detector = ModelDetector()
    detected = detector.fetch_model_info(model_id)
    if not detected:
        console.print(f"[red]Could not fetch model info for {model_id}[/red]")
        return 1

    profiler = ModelProfiler()
    result = profiler.profile(detected)

    if args.json_output:
        print(json.dumps({
            "model_id": model_id,
            "tag": result.profile.tag,
            "family": result.profile.family,
            "math_intensity": result.profile.math_intensity,
            "precision": result.profile.precision,
            "is_memory_bound": result.profile.is_memory_bound,
            "typical_util_min": result.profile.typical_util_min,
            "typical_util_max": result.profile.typical_util_max,
            "confidence": result.confidence,
            "inferred_from": result.inferred_from,
            "reasoning": result.reasoning,
        }, indent=2))
        return 0

    console.print(f"[bold]Profile: {model_id}[/bold]\n")
    console.print(f"  Tag:            [cyan]{result.profile.tag}[/cyan]")
    console.print(f"  Family:         [green]{result.profile.family}[/green]")
    console.print(f"  Math intensity: {result.profile.math_intensity:.1f} FLOP/byte")
    console.print(f"  Precision:      {result.profile.precision}")
    console.print(f"  Memory-bound:   {'Yes' if result.profile.is_memory_bound else 'No'}")
    console.print(f"  Utilization:    {result.profile.typical_util_min}-{result.profile.typical_util_max}%")
    console.print(f"  Confidence:     {result.confidence:.0%}")
    console.print(f"  Inferred from:  {result.inferred_from}")
    console.print(f"\n  [dim]Reasoning:[/dim]")
    for part in result.reasoning.split("; "):
        console.print(f"    - {part}")

    return 0


def _cmd_warm_start(pipeline, args) -> int:
    console = Console()
    console.print("[bold]Warming up bandit with model intelligence data...[/bold]\n")

    count = pipeline.warm_start_bandit()
    if count > 0:
        console.print(f"[green]Generated {count} synthetic experience tuples[/green]")
    else:
        console.print("[yellow]No models available for warm-start. Run 'scan' first.[/yellow]")

    return 0


def _cmd_quantize(pipeline, args) -> int:
    console = Console()
    model_id = args.model

    console.print(f"[bold]Analyzing quantization variants for {model_id}...[/bold]\n")

    entry = pipeline.run_single(model_id)
    if not entry:
        console.print(f"[red]Could not profile model: {model_id}[/red]")
        return 1

    from intelligence.quantization import QuantizationAdvisor

    advisor = QuantizationAdvisor()

    if args.gpu:
        rec = advisor.recommend_per_gpu(entry.profile, args.gpu, entry.parameter_count)
        if not rec:
            console.print(f"[yellow]No quantization recommendation for {args.gpu}[/yellow]")
            return 0

        if args.json_output:
            print(json.dumps({
                "model": model_id,
                "gpu": args.gpu,
                "recommended_variant": rec.variant.name,
                "model_size_gb": rec.model_size_gb,
                "memory_reduction_pct": rec.memory_reduction_pct,
                "throughput_change_pct": rec.estimated_throughput_change_pct,
                "quality_impact": rec.estimated_quality_impact,
            }, indent=2))
            return 0

        console.print(f"  GPU: [cyan]{args.gpu}[/cyan]")
        console.print(f"  Recommended: [green]{rec.variant.name}[/green] ({rec.variant.format_name})")
        console.print(f"  Size: {rec.model_size_gb:.1f} GB ({rec.memory_reduction_pct:.0f}% reduction)")
        console.print(f"  Throughput: {rec.estimated_throughput_change_pct:+.0f}%")
        console.print(f"  Quality: {rec.estimated_quality_impact}")
        if rec.warnings:
            for w in rec.warnings:
                console.print(f"  [yellow]Warning: {w}[/yellow]")
        return 0

    result = advisor.analyze(entry.profile, entry.parameter_count)

    if args.json_output:
        print(json.dumps({
            "model": model_id,
            "sweet_spot": result.sweet_spot.variant.name if result.sweet_spot else None,
            "variants": [
                {
                    "variant": v.variant.name,
                    "format": v.variant.format_name,
                    "precision": v.variant.precision,
                    "size_gb": v.model_size_gb,
                    "memory_reduction_pct": v.memory_reduction_pct,
                    "throughput_change_pct": v.estimated_throughput_change_pct,
                    "quality_impact": v.estimated_quality_impact,
                    "best_gpu": v.gpu_rankings[0]["gpu_name"] if v.gpu_rankings else None,
                    "fits_on_count": len(v.fits_on_gpus),
                    "warnings": v.warnings,
                }
                for v in result.variants
            ],
        }, indent=2))
        return 0

    sweet = result.sweet_spot.variant.name if result.sweet_spot else "N/A"
    table = Table(title=f"Quantization Variants for {entry.tag} (sweet spot: {sweet})")
    table.add_column("Variant", style="cyan")
    table.add_column("Format")
    table.add_column("Size (GB)", justify="right")
    table.add_column("Mem Reduction", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Quality", style="green")
    table.add_column("Best GPU", style="yellow")
    table.add_column("Fits On", justify="right")

    for v in result.variants:
        best_gpu = v.gpu_rankings[0]["gpu_name"] if v.gpu_rankings else "—"
        style = "bold" if result.sweet_spot and v.variant.name == result.sweet_spot.variant.name else ""
        table.add_row(
            v.variant.name,
            v.variant.format_name,
            f"{v.model_size_gb:.1f}",
            f"{v.memory_reduction_pct:.0f}%",
            f"{v.estimated_throughput_change_pct:+.0f}%",
            v.estimated_quality_impact,
            best_gpu,
            str(len(v.fits_on_gpus)),
            style=style,
        )

    console.print(table)

    if result.sweet_spot and result.sweet_spot.warnings:
        console.print()
        for w in result.sweet_spot.warnings:
            console.print(f"[yellow]  Warning: {w}[/yellow]")

    return 0


def _cmd_prices(data_dir, args) -> int:
    console = Console()
    from intelligence.pricing import GPUPricingTracker

    tracker = GPUPricingTracker(data_dir=data_dir)

    if args.update_from:
        count = tracker.update_from_json(Path(args.update_from))
        console.print(f"[green]Updated {count} GPU prices from {args.update_from}[/green]\n")

    sources = tracker.get_all_sources()

    if args.json_output:
        print(json.dumps({
            gpu: {
                "on_demand_rate": src.on_demand_rate,
                "spot_rate": src.spot_rate,
                "provider": src.provider,
            }
            for gpu, src in sources.items()
        }, indent=2))
        return 0

    table = Table(title=f"GPU Pricing ({len(sources)} GPUs)")
    table.add_column("GPU", style="cyan")
    table.add_column("$/hr", justify="right", style="green")
    table.add_column("Spot $/hr", justify="right", style="yellow")
    table.add_column("Provider")

    for gpu_name in sorted(sources.keys()):
        src = sources[gpu_name]
        spot = f"${src.spot_rate:.2f}" if src.spot_rate else "—"
        table.add_row(
            gpu_name,
            f"${src.on_demand_rate:.2f}",
            spot,
            src.provider,
        )

    console.print(table)
    return 0


def _cmd_best_value(pipeline, data_dir, args) -> int:
    console = Console()
    model_id = args.model

    from intelligence.pricing import GPUPricingTracker
    from efficiency.gpu_specs import MODEL_PROFILES

    tracker = GPUPricingTracker(data_dir=data_dir)

    # Try local profile first, then fetch
    profile = MODEL_PROFILES.get(model_id)
    if not profile:
        entry = pipeline.run_single(model_id)
        if not entry:
            console.print(f"[red]Could not profile model: {model_id}[/red]")
            return 1
        profile = entry.profile

    console.print(f"[bold]Best $/TFLOP GPUs for {profile.tag}[/bold]\n")

    results = tracker.compute_price_performance(profile, top_n=args.top)

    if args.budget:
        results = [r for r in results if r.on_demand_rate <= args.budget]

    if args.json_output:
        print(json.dumps([
            {
                "gpu": r.gpu_name,
                "on_demand_rate": r.on_demand_rate,
                "spot_rate": r.spot_rate,
                "effective_tflops": r.effective_tflops,
                "dollars_per_tflop_hr": r.dollars_per_tflop_hr,
                "value_score": r.value_score,
                "is_best_value": r.is_best_value,
            }
            for r in results
        ], indent=2))
        return 0

    if not results:
        console.print("[yellow]No GPUs found within budget[/yellow]")
        return 0

    table = Table(title=f"Price-Performance Ranking ({profile.tag})")
    table.add_column("#", justify="right", style="dim")
    table.add_column("GPU", style="cyan")
    table.add_column("$/hr", justify="right", style="yellow")
    table.add_column("Eff. TFLOPS", justify="right")
    table.add_column("$/TFLOP-hr", justify="right", style="green")
    table.add_column("Value Score", justify="right")

    for i, r in enumerate(results, 1):
        style = "bold" if r.is_best_value else ""
        table.add_row(
            str(i),
            r.gpu_name,
            f"${r.on_demand_rate:.2f}",
            f"{r.effective_tflops:.1f}",
            f"${r.dollars_per_tflop_hr:.4f}",
            f"{r.value_score:.0f}",
            style=style,
        )

    console.print(table)
    return 0
