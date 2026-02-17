"""
GPU Energy Agent - Main Entry Point

Monitors GPU energy consumption with minimal overhead.
"""

import argparse
import csv
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from collector import GPUCollector, CSV_HEADER

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from uploader import MetricsUploader
    from config import UPLOAD_ENABLED, UPLOAD_INTERVAL, API_KEY, SCHEDULER_POLL_INTERVAL
    UPLOADER_AVAILABLE = True
except ImportError:
    UPLOADER_AVAILABLE = False
    UPLOAD_ENABLED = False
    SCHEDULER_POLL_INTERVAL = 30

try:
    from schedulers import detect_scheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False


class EnergyAgent:
    """Main agent class for GPU energy monitoring"""

    def __init__(
        self,
        interval: float = 5.0,
        output_csv: Optional[str] = None,
        duration: Optional[int] = None,
        quiet: bool = False
    ):
        """
        Initialize energy agent.

        Args:
            interval: Sampling interval in seconds (default: 5.0)
            output_csv: Path to CSV output file (optional)
            duration: Run duration in seconds (None = infinite)
            quiet: Suppress console output
        """
        self.interval = interval
        self.output_csv = output_csv
        self.duration = duration
        self.quiet = quiet

        self.collector = None
        self.running = False
        self.sample_count = 0
        self.total_energy = {}  # gpu_index -> total kWh

        # Initialize uploader if API key is configured
        self.uploader = None
        self.last_upload_time = time.time()
        if UPLOADER_AVAILABLE and UPLOAD_ENABLED and API_KEY:
            self.uploader = MetricsUploader()
            if not quiet:
                print("✅ API upload enabled")
        elif not quiet and UPLOADER_AVAILABLE:
            print("⚠️  API upload disabled (no API key)")

        # Initialize scheduler adapter for job attribution
        self.scheduler = None
        self.last_scheduler_poll = 0.0
        if SCHEDULER_AVAILABLE:
            self.scheduler = detect_scheduler()
            if not quiet:
                print(f"🔗 Scheduler: {self.scheduler.name}")
        elif not quiet:
            print("⚠️  Scheduler integration unavailable")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Rich console for pretty output
        if RICH_AVAILABLE and not quiet:
            self.console = Console()
        else:
            self.console = None

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n\n🛑 Shutting down agent...")
        self.running = False

    def run(self):
        """Main agent loop"""
        self.running = True
        start_time = time.time()

        # Initialize collector
        try:
            self.collector = GPUCollector(collect_clocks=False)
            gpu_count = self.collector.get_gpu_count()

            if not self.quiet:
                self._print_header(gpu_count)

            # Initialize total energy tracking
            for i in range(gpu_count):
                self.total_energy[i] = 0.0

            # Retry failed uploads from previous runs
            if self.uploader:
                retried = self.uploader.retry_failed_uploads()
                if retried > 0 and not self.quiet:
                    print(f"🔄 Retried {retried} previously failed metrics")

        except Exception as e:
            print(f"❌ Failed to initialize collector: {e}")
            return 1

        # Setup CSV output
        csv_file = None
        csv_writer = None
        if self.output_csv:
            csv_path = Path(self.output_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(CSV_HEADER)
            print(f"📁 Writing metrics to: {self.output_csv}")

        # Main collection loop
        try:
            while self.running:
                loop_start = time.time()

                # Poll scheduler for job attribution (every SCHEDULER_POLL_INTERVAL)
                if self.scheduler and (loop_start - self.last_scheduler_poll >= SCHEDULER_POLL_INTERVAL):
                    try:
                        self.scheduler.discover_jobs()
                        self.last_scheduler_poll = loop_start
                    except Exception as e:
                        if not self.quiet:
                            print(f"⚠️  Scheduler poll failed: {e}")

                # Collect metrics
                metrics = self.collector.collect()
                self.sample_count += 1

                # Enrich metrics with job attribution
                if self.scheduler:
                    for m in metrics:
                        job = self.scheduler.gpu_to_job(m.gpu_index)
                        if job:
                            m.job_id = job.job_id
                            m.team_id = job.team_id
                            m.model_tag = job.model_tag
                            m.scheduler_source = job.scheduler_source

                # Update total energy
                for m in metrics:
                    if m.energy_delta_j:
                        self.total_energy[m.gpu_index] += m.energy_delta_j / 3_600_000  # J -> kWh

                # Add metrics to uploader buffer
                if self.uploader:
                    metric_dicts = [m.to_dict() for m in metrics]
                    self.uploader.add_metrics(metric_dicts)

                    # Flush buffer periodically
                    if time.time() - self.last_upload_time >= UPLOAD_INTERVAL:
                        uploaded = self.uploader.flush()
                        self.last_upload_time = time.time()

                # Write to CSV
                if csv_writer:
                    for m in metrics:
                        csv_writer.writerow(m.to_csv_row())
                    csv_file.flush()

                # Display metrics
                if not self.quiet:
                    self._display_metrics(metrics)

                # Check duration
                if self.duration and (time.time() - start_time) >= self.duration:
                    print(f"\n⏱️  Duration limit reached ({self.duration}s)")
                    break

                # Sleep until next sample
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            print(f"\n❌ Error during collection: {e}")
            return 1

        finally:
            # Flush remaining metrics to API
            if self.uploader and len(self.uploader.buffer) > 0:
                print(f"\n📤 Uploading {len(self.uploader.buffer)} remaining metrics...")
                self.uploader.flush()

            # Cleanup
            if csv_file:
                csv_file.close()
                print(f"✅ CSV file written: {self.output_csv}")

            if self.collector:
                self.collector.shutdown()

            self._print_summary(time.time() - start_time)

        return 0

    def _print_header(self, gpu_count: int):
        """Print startup header"""
        if self.console:
            self.console.print("\n[bold green]GPU Energy Agent v0.1.0[/bold green]")
            self.console.print(f"📊 Monitoring [cyan]{gpu_count}[/cyan] GPUs")
            self.console.print(f"⏱️  Sampling interval: [cyan]{self.interval}s[/cyan]")
            if self.duration:
                self.console.print(f"⏱️  Duration: [cyan]{self.duration}s[/cyan]")
            self.console.print()
        else:
            print(f"\nGPU Energy Agent v0.1.0")
            print(f"Monitoring {gpu_count} GPUs")
            print(f"Sampling interval: {self.interval}s\n")

    def _display_metrics(self, metrics):
        """Display current metrics to console"""
        if self.console and RICH_AVAILABLE:
            # Rich table output
            table = Table(title=f"Sample #{self.sample_count}")
            table.add_column("GPU", style="cyan")
            table.add_column("Power", justify="right")
            table.add_column("Util", justify="right")
            table.add_column("Temp", justify="right")
            table.add_column("Energy Δ", justify="right")
            table.add_column("Total kWh", justify="right")

            for m in metrics:
                energy_str = f"{m.energy_delta_j:.1f}J" if m.energy_delta_j else "N/A"
                total_kwh = self.total_energy.get(m.gpu_index, 0)

                table.add_row(
                    f"GPU {m.gpu_index}",
                    f"{m.power_draw_w:.1f}W",
                    f"{m.utilization_gpu_pct}%",
                    f"{m.temperature_c}°C",
                    energy_str,
                    f"{total_kwh:.4f}"
                )

            self.console.clear()
            self.console.print(table)

        else:
            # Simple text output
            print(f"\n[Sample #{self.sample_count}] {datetime.now().strftime('%H:%M:%S')}")
            for m in metrics:
                energy_str = f"{m.energy_delta_j:.1f}J" if m.energy_delta_j else "N/A"
                total_kwh = self.total_energy.get(m.gpu_index, 0)
                print(f"  GPU {m.gpu_index} | "
                      f"Power: {m.power_draw_w:6.1f}W | "
                      f"Util: {m.utilization_gpu_pct:3d}% | "
                      f"Temp: {m.temperature_c:3d}°C | "
                      f"ΔE: {energy_str:>8} | "
                      f"Total: {total_kwh:.4f} kWh")

    def _print_summary(self, runtime: float):
        """Print final summary"""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Runtime:        {runtime:.1f}s")
        print(f"Samples:        {self.sample_count}")
        print(f"Sample interval: {self.interval}s")

        # Uploader stats
        if self.uploader:
            status = self.uploader.get_status()
            print(f"\nAPI Upload:")
            print(f"  Status:         {'✅ Enabled' if status['has_api_key'] else '❌ Disabled'}")
            print(f"  Buffer:         {status['buffer_size']} metrics")
            if status['failed_metrics_count'] > 0:
                print(f"  Failed:         {status['failed_metrics_count']} metrics (will retry next run)")

        print(f"\nTotal Energy Consumed:")

        total_all = 0
        for gpu_idx, kwh in sorted(self.total_energy.items()):
            print(f"  GPU {gpu_idx}: {kwh:.6f} kWh ({kwh * 3_600_000:.1f} J)")
            total_all += kwh

        print(f"  TOTAL:  {total_all:.6f} kWh (${total_all * 0.12:.4f} @ $0.12/kWh)")
        print("="*60 + "\n")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='GPU Energy Monitoring Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor with default settings (5s interval)
  python main.py

  # Fast sampling (1s intervals)
  python main.py --interval 1

  # Run for 5 minutes, save to CSV
  python main.py --duration 300 --output data/test.csv

  # Quiet mode (no console output)
  python main.py --quiet --output data/metrics.csv
        """
    )

    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=5.0,
        help='Sampling interval in seconds (default: 5.0)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output CSV file path'
    )

    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=None,
        help='Run duration in seconds (default: infinite)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress console output'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.interval < 0.1:
        print("Error: Interval must be >= 0.1s")
        return 1

    if args.interval > 3600:
        print("Warning: Interval > 1 hour may miss important events")

    # Run agent
    agent = EnergyAgent(
        interval=args.interval,
        output_csv=args.output,
        duration=args.duration,
        quiet=args.quiet
    )

    return agent.run()


if __name__ == '__main__':
    sys.exit(main())
