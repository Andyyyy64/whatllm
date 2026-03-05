"""Rich output formatting for CLI display."""

from __future__ import annotations

import json
import re
from datetime import datetime
from math import log10

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from whichllm.engine.quantization import effective_quant_type, estimate_weight_bytes
from whichllm.engine.types import CompatibilityResult
from whichllm.hardware.types import HardwareInfo

console = Console()


def _format_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    elif b >= 1024**2:
        return f"{b / 1024**2:.0f} MB"
    return f"{b / 1024:.0f} KB"


def _format_params(count: int) -> str:
    """Format parameter count."""
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.0f}M"
    return str(count)


def _format_downloads(downloads: int) -> str:
    """Format download count for compact table display."""
    if downloads >= 1_000_000:
        return f"{downloads / 1_000_000:.1f}M"
    if downloads >= 1_000:
        return f"{downloads / 1_000:.1f}K"
    return str(downloads)


def _format_published_at(value: str | None) -> str:
    """Format published datetime into YYYY-MM-DD."""
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return value[:10] if len(value) >= 10 else value


def _parse_published_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _lerp_channel(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def _blend_hex(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    r = _lerp_channel(a[0], b[0], t)
    g = _lerp_channel(a[1], b[1], t)
    bch = _lerp_channel(a[2], b[2], t)
    return f"#{r:02x}{g:02x}{bch:02x}"


def _downloads_style(downloads: int, min_log: float, max_log: float) -> str:
    if downloads <= 0:
        return "grey50"
    dlog = log10(max(downloads, 1))
    span = max(max_log - min_log, 1e-6)
    t = (dlog - min_log) / span
    return _blend_hex((145, 80, 80), (55, 190, 120), t)


def _published_style(
    published: datetime | None,
    oldest_ts: float | None,
    newest_ts: float | None,
) -> str:
    if published is None or oldest_ts is None or newest_ts is None:
        return "grey50"
    pts = published.timestamp()
    span = max(newest_ts - oldest_ts, 1e-6)
    t = (pts - oldest_ts) / span
    return _blend_hex((190, 85, 85), (80, 190, 110), t)


def _detect_specializations(model_id: str) -> list[str]:
    """Detect task-specialized model hints from repository name."""
    lower = model_id.lower()
    tags: list[str] = []
    if re.search(r"(coder|codegen|starcoder|program|coding)", lower):
        tags.append("coding")
    if re.search(r"(^|[-_/])(vl|vision|multimodal|llava|image)([-_/]|$)", lower):
        tags.append("vision")
    if re.search(r"(^|[-_/])math([-_/]|$)", lower):
        tags.append("math")
    return tags


def _top_pick_confidence(results: list[CompatibilityResult]) -> tuple[str, str]:
    """Return confidence level and explanation for top pick."""
    top = results[0]
    gap = (top.quality_score - results[1].quality_score) if len(results) > 1 else 999.0
    fit_note = ""
    if top.fit_type == "partial_offload":
        fit_note = ", partial offload"
    elif top.fit_type == "cpu_only":
        fit_note = ", CPU-only"

    if top.benchmark_status == "none":
        return "Low", f"no benchmark data, gap +{gap:.1f}{fit_note}"
    if top.benchmark_status == "estimated":
        if gap >= 2.0:
            return "Medium", f"estimated benchmark, gap +{gap:.1f}{fit_note}"
        return "Low", f"estimated benchmark, gap +{gap:.1f}{fit_note}"
    # direct benchmark
    if gap >= 2.5:
        confidence = "High"
        reason = f"direct benchmark, gap +{gap:.1f}{fit_note}"
    elif gap >= 1.0:
        confidence = "Medium"
        reason = f"direct benchmark, gap +{gap:.1f}{fit_note}"
    else:
        confidence = "Low"
        reason = f"direct benchmark but very close (+{gap:.1f}){fit_note}"

    # オフロード/CPU-onlyの1位は実運用で不確実性が高いため信頼度を1段階下げる
    if top.fit_type != "full_gpu":
        if confidence == "High":
            confidence = "Medium"
        elif confidence == "Medium":
            confidence = "Low"
    return confidence, reason


def display_hardware(hw: HardwareInfo) -> None:
    """Display hardware information panel."""
    lines: list[str] = []

    # GPUs
    if hw.gpus:
        for i, gpu in enumerate(hw.gpus):
            vram = _format_bytes(gpu.vram_bytes)
            bw = f"{gpu.memory_bandwidth_gbps:.0f} GB/s" if gpu.memory_bandwidth_gbps else "N/A"
            cc = (
                f"CC {gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
                if gpu.compute_capability
                else ""
            )
            extra = []
            if cc:
                extra.append(cc)
            if gpu.cuda_version:
                extra.append(f"CUDA {gpu.cuda_version}")
            if gpu.rocm_version:
                extra.append(f"ROCm {gpu.rocm_version}")
            extra_str = f" ({', '.join(extra)})" if extra else ""
            lines.append(f"[bold green]GPU {i}:[/] {gpu.name} — {vram}{extra_str} — BW: {bw}")
    else:
        lines.append("[yellow]No GPU detected[/] — CPU-only mode")

    # CPU
    avx_flags = []
    if hw.has_avx2:
        avx_flags.append("AVX2")
    if hw.has_avx512:
        avx_flags.append("AVX-512")
    avx_str = f" ({', '.join(avx_flags)})" if avx_flags else ""
    lines.append(f"[bold blue]CPU:[/] {hw.cpu_name} — {hw.cpu_cores} cores{avx_str}")

    # Memory
    lines.append(f"[bold blue]RAM:[/] {_format_bytes(hw.ram_bytes)}")
    lines.append(f"[bold blue]Disk free:[/] {_format_bytes(hw.disk_free_bytes)}")
    lines.append(f"[bold blue]OS:[/] {hw.os}")

    panel = Panel("\n".join(lines), title="[bold]Hardware Info[/]", border_style="blue")
    console.print(panel)


def display_ranking(
    results: list[CompatibilityResult],
    *,
    has_gpu: bool = True,
    show_status: bool = False,
) -> None:
    """Display ranked model table."""
    if not results:
        console.print("[yellow]No compatible models found for your hardware.[/]")
        return

    mem_label = "VRAM" if has_gpu else "RAM"

    table = Table(title="Recommended Models", show_lines=True)
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Model", style="cyan", min_width=14, overflow="fold")
    table.add_column("Params", justify="right", width=6)
    table.add_column("Quant", justify="center", width=6)
    if show_status:
        table.add_column(mem_label, justify="right", width=8)
        table.add_column("Speed", justify="right", width=8)
        table.add_column("Fit", justify="center", width=7)
    else:
        table.add_column("Published", justify="center", width=10)
        table.add_column("Downloads", justify="right", width=9)
    table.add_column("Score", justify="right", width=5)
    table.add_column("License", width=8)

    download_logs = [log10(max(r.model.downloads, 1)) for r in results if r.model.downloads > 0]
    min_download_log = min(download_logs) if download_logs else 0.0
    max_download_log = max(download_logs) if download_logs else 1.0
    published_dates = [_parse_published_at(r.model.published_at) for r in results]
    published_valid = [d for d in published_dates if d is not None]
    oldest_ts = min((d.timestamp() for d in published_valid), default=None)
    newest_ts = max((d.timestamp() for d in published_valid), default=None)

    for i, r in enumerate(results, 1):
        quant = effective_quant_type(r.model, r.gguf_variant)
        vram_str = _format_bytes(r.vram_required_bytes)
        speed_str = f"{r.estimated_tok_per_sec:.1f} tok/s" if r.estimated_tok_per_sec else "N/A"

        # Score with benchmark status indicator
        score_val = f"{r.quality_score:.1f}"
        if r.benchmark_status == "none":
            score_str = f"[red]{score_val} ?[/red]"
        elif r.benchmark_status == "estimated":
            score_str = f"[yellow]{score_val} ~[/yellow]"
        else:
            score_str = f"[green]{score_val}[/green]"

        fit_style = {
            "full_gpu": "[green]Full GPU[/]",
            "partial_offload": "[yellow]Partial[/]",
            "cpu_only": "[red]CPU only[/]",
        }
        fit_str = fit_style.get(r.fit_type, r.fit_type)
        published_dt = _parse_published_at(r.model.published_at)
        published_str = Text(
            _format_published_at(r.model.published_at),
            style=_published_style(published_dt, oldest_ts, newest_ts),
        )
        downloads_str = Text(
            _format_downloads(r.model.downloads),
            style=_downloads_style(r.model.downloads, min_download_log, max_download_log),
        )

        params_str = _format_params(r.model.parameter_count)
        if r.model.is_moe and r.model.parameter_count_active:
            params_str += f" ({_format_params(r.model.parameter_count_active)}a)"

        license_str = r.model.license or "—"

        model_link = Text(r.model.id, style="cyan")
        model_link.stylize(f"link https://huggingface.co/{r.model.id}")

        row_cells = [
            str(i),
            model_link,
            params_str,
            quant,
        ]
        if show_status:
            row_cells.extend([vram_str, speed_str, fit_str])
        else:
            row_cells.extend([published_str, downloads_str])
        row_cells.extend([score_str, license_str])
        table.add_row(*row_cells)

    console.print(table)

    # Score legend
    has_estimated = any(r.benchmark_status == "estimated" for r in results)
    has_none = any(r.benchmark_status == "none" for r in results)
    if has_estimated or has_none:
        parts = []
        if has_estimated:
            parts.append("[yellow]Estimated / ~[/yellow] = inferred from model line")
        if has_none:
            parts.append("[red]None / ?[/red] = no benchmark data")
        console.print(f"  [dim]Score:[/dim]  {',  '.join(parts)}")

    has_direct = any(r.benchmark_status == "direct" for r in results)
    if not has_direct:
        console.print(
            "  [red]No confirmed winner:[/] direct benchmark data is missing for current candidates."
        )

    confidence, reason = _top_pick_confidence(results)
    confidence_style = {
        "High": "green",
        "Medium": "yellow",
        "Low": "red",
    }[confidence]
    console.print(
        f"  Top pick confidence: [{confidence_style}]{confidence}[/{confidence_style}] ({reason})"
    )

    # 上位が僅差なら「断定しすぎない」ための注意を表示する
    if len(results) >= 2:
        gap = results[0].quality_score - results[1].quality_score
        if gap < 1.5:
            console.print(
                f"  [yellow]Note:[/] Top candidates are very close (#{1} vs #{2}: {gap:.1f} pts)."
            )

    # 上位に根拠が弱い候補がある場合は目立つ注意を出す
    weak_top = [idx + 1 for idx, r in enumerate(results[:3]) if r.benchmark_status != "direct"]
    if weak_top:
        joined = ", ".join(f"#{i}" for i in weak_top)
        console.print(f"  [yellow]Caution:[/] Weaker benchmark evidence in top ranks: {joined}")

    specialized: list[str] = []
    for idx, r in enumerate(results[:10], 1):
        tags = _detect_specializations(r.model.id)
        if tags:
            joined_tags = "/".join(tags)
            specialized.append(f"#{idx} {joined_tags}")
    if specialized:
        console.print(
            "  [yellow]Task hint:[/] Specialized models detected in ranking: "
            + ", ".join(specialized)
        )

    # Show warnings for top results
    for i, r in enumerate(results[:3], 1):
        if r.warnings:
            for w in r.warnings:
                console.print(f"  [yellow]Warning #{i} {r.model.name}:[/] {w}")


def display_json(results: list[CompatibilityResult], hardware: HardwareInfo) -> None:
    """Output results as JSON."""
    output = {
        "hardware": {
            "gpus": [
                {
                    "name": g.name,
                    "vendor": g.vendor,
                    "vram_bytes": g.vram_bytes,
                    "memory_bandwidth_gbps": g.memory_bandwidth_gbps,
                }
                for g in hardware.gpus
            ],
            "cpu": hardware.cpu_name,
            "cpu_cores": hardware.cpu_cores,
            "ram_bytes": hardware.ram_bytes,
            "os": hardware.os,
        },
        "models": [
            {
                "rank": i,
                "model_id": r.model.id,
                "parameter_count": r.model.parameter_count,
                "published_at": r.model.published_at,
                "downloads": r.model.downloads,
                "quant_type": effective_quant_type(r.model, r.gguf_variant),
                "file_size_bytes": (
                    r.gguf_variant.file_size_bytes
                    if r.gguf_variant
                    else estimate_weight_bytes(r.model, None)
                ),
                "vram_required_bytes": r.vram_required_bytes,
                "estimated_tok_per_sec": r.estimated_tok_per_sec,
                "quality_score": round(r.quality_score, 2),
                "benchmark_status": r.benchmark_status,
                "fit_type": r.fit_type,
                "can_run": r.can_run,
                "warnings": r.warnings,
                "license": r.model.license,
            }
            for i, r in enumerate(results, 1)
        ],
    }
    console.print_json(json.dumps(output, ensure_ascii=False))
