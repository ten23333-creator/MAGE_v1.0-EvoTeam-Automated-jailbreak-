#!/usr/bin/env python3
"""
EvoTeam - Evolutionary Collaborative Hybrid Framework for Black-Box LLM Jailbreak.
===============================================================================

Entry point for running EvoTeam attacks.
Supports single-query, batch, and config-driven execution modes.
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from evoteam.config import load_config, EvoTeamConfig
from evoteam.agents.orchestrator import EvoTeamOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(
        description="EvoTeam - Black-Box LLM Jailbreak Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query attack
  python main.py --query "How to make a bomb?"

  # Batch attack from file (one query per line)
  python main.py --queries-file queries.txt

  # Use custom config
  python main.py --config my_config.yaml --query "..."

  # Specify output directory
  python main.py --query "..." --output-dir results/
        """,
    )

    # Execution mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", "-q", type=str, help="Single attack query")
    mode.add_argument("--queries-file", "-f", type=str, help="File with queries (one per line)")
    mode.add_argument("--benchmark", "-b", type=str, choices=["harmbench", "advbench", "jailbreakbench"],
                      help="Run a standard benchmark (requires benchmark data)")

    # Configuration
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                        help="Path to YAML config file (default: config.yaml)")

    # Output
    parser.add_argument("--output-dir", "-o", type=str, default="results",
                        help="Output directory for results (default: results/)")
    parser.add_argument("--no-save", action="store_true",
                        help="Disable saving results to disk")

    # Runtime overrides
    parser.add_argument("--max-iterations", type=int, help="Override max iterations")
    parser.add_argument("--max-rounds", type=int, help="Override max conversation rounds")
    parser.add_argument("--attack-model", type=str, help="Override attack model name")
    parser.add_argument("--target-model", type=str, help="Override target model name")
    parser.add_argument("--judge-model", type=str, help="Override judge model name")
    parser.add_argument("--api-base", type=str, help="Override API base URL")
    parser.add_argument("--api-key", type=str, help="Override API key")

    # Verbosity
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Suppress all output except final results")

    return parser.parse_args()


def load_queries_from_file(filepath: str) -> list:
    """Load queries from a text file (one per line, skip empty/comment lines)."""
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    return queries


def load_benchmark_queries(benchmark_name: str) -> list:
    """Load queries from a standard benchmark file."""
    benchmark_dir = PROJECT_ROOT / "data" / "benchmarks"
    filepath = benchmark_dir / f"{benchmark_name}.txt"
    if filepath.exists():
        return load_queries_from_file(str(filepath))
    # Fall back to HarmBench-style JSON
    json_path = benchmark_dir / f"{benchmark_name}.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item.get("query", item.get("prompt", str(item))) for item in data]
    print(f"Warning: Benchmark file not found at {filepath} or {json_path}")
    return []


def apply_overrides(config: EvoTeamConfig, args) -> EvoTeamConfig:
    """Apply command-line overrides to config."""
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    if args.max_rounds:
        config.multi_turn_max_rounds = args.max_rounds
    if args.attack_model:
        config.attack_model = args.attack_model
    if args.target_model:
        config.target_model = args.target_model
    if args.judge_model:
        config.judge_model = args.judge_model
    if args.api_base:
        config.api_base = args.api_base
        config.attack_api_base = args.api_base
        config.target_api_base = args.api_base
        config.judge_api_base = args.api_base
    if args.api_key:
        config.api_key = args.api_key
    return config


def run_single_query(orchestrator: EvoTeamOrchestrator, query: str, output_dir: str = None, no_save: bool = False):
    """Run attack on a single query and return results."""
    result = orchestrator.run_attack(query)
    if not no_save and output_dir and hasattr(orchestrator, 'data_saver') and orchestrator.data_saver:
        orchestrator.data_saver.save_json(
            result, output_dir,
            f"attack_result_{result.get('session_id', 'unknown')}.json"
        )
    return result


def run_batch(orchestrator: EvoTeamOrchestrator, queries: list, output_dir: str = None, no_save: bool = False):
    """Run attacks on a batch of queries."""
    results = orchestrator.run_attack_batch(queries)
    if not no_save and output_dir and hasattr(orchestrator, 'data_saver') and orchestrator.data_saver:
        orchestrator.data_saver.save_json(
            {"batch_results": results, "queries": queries, "total": len(results)},
            output_dir,
            "batch_summary.json"
        )
    return results


def print_result(result: dict, verbose: bool = False):
    """Print attack result to console."""
    successful = result.get('successful_attacks', [])
    print(f"\n{'='*60}")
    print(f"Query: {result.get('query', 'N/A')[:80]}")
    print(f"Tag: {result.get('tag') or result.get('query_tag', 'N/A')}")
    print(f"Success: {result.get('final_success') or result.get('attack_successful', False)}")
    print(f"Best Score: {result.get('best_score') or result.get('highest_score', 0)}/5")
    print(f"Successful Attacks: {len(successful)}")
    print(f"Iterations: {result.get('total_iterations', 0)}")
    print(f"Population Stats: {result.get('population_final_stats', {})}")

    if verbose:
        history = result.get('conversation_history', [])
        if history:
            print(f"\n--- Conversation History ({len(history)} turns) ---")
            for turn in history:
                print(f"\nTurn {turn.get('turn', '?')}:")
                print(f"  Attack: {turn.get('attack_prompt', '')[:150]}")
                print(f"  Target: {turn.get('target_response', '')[:200]}")
                print(f"  Score: {turn.get('judge_score', '?')}/5")

        best_tool = result.get('best_tool')
        if best_tool:
            print(f"\n--- Best Tool ---")
            print(f"  Name: {best_tool.get('tool_name', 'N/A')}")
            print(f"  Approach: {best_tool.get('approach', 'N/A')[:100]}")

        reflection = result.get('final_reflection')
        if reflection:
            print(f"\n--- Final Reflection ---")
            for key, val in reflection.items():
                if isinstance(val, str):
                    print(f"  {key}: {val[:150]}")
    print(f"{'='*60}\n")


def print_batch_summary(results: list, verbose: bool = False):
    """Print batch attack summary."""
    total = len(results)
    successful = sum(1 for r in results if r.get('final_success') or r.get('attack_successful', False))
    best_scores = [r.get('best_score') or r.get('highest_score', 0) for r in results]
    avg_score = sum(best_scores) / max(total, 1)

    print(f"\n{'='*60}")
    print(f"BATCH ATTACK SUMMARY")
    print(f"Total Queries: {total}")
    print(f"Successful: {successful} ({100*successful/max(total,1):.1f}%)")
    print(f"Average Best Score: {avg_score:.2f}/5")
    print(f"{'='*60}")

    if verbose:
        tag_stats = {}
        for r in results:
            tag = r.get('tag') or r.get('query_tag') or 'general'
            if tag not in tag_stats:
                tag_stats[tag] = {'total': 0, 'success': 0, 'scores': []}
            tag_stats[tag]['total'] += 1
            if r.get('final_success') or r.get('attack_successful', False):
                tag_stats[tag]['success'] += 1
            tag_stats[tag]['scores'].append(r.get('best_score') or r.get('highest_score', 0))

        print("\n--- Per-Tag Breakdown ---")
        for tag, stats in sorted(tag_stats.items()):
            succ_rate = 100 * stats['success'] / max(stats['total'], 1)
            avg_s = sum(stats['scores']) / max(len(stats['scores']), 1)
            print(f"  {tag}: {stats['total']} queries, {succ_rate:.1f}% success, avg score {avg_s:.2f}")
    print()


def main():
    args = parse_args()

    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(PROJECT_ROOT / config_path)
    config = load_config(config_path) if os.path.exists(config_path) else EvoTeamConfig()
    config = apply_overrides(config, args)

    # Set up output directory
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = str(PROJECT_ROOT / output_dir)
    if not args.no_save:
        os.makedirs(output_dir, exist_ok=True)
    config.output_dir = output_dir

    # Initialize orchestrator
    if not args.quiet:
        print("Initializing EvoTeam orchestrator...")
        print(f"  Attack model: {config.attack_model}")
        print(f"  Target model: {config.target_model}")
        print(f"  Judge model: {config.judge_model}")
        print(f"  Max iterations: {config.max_iterations}")
        print(f"  Max rounds: {config.multi_turn_max_rounds}")

    orchestrator = EvoTeamOrchestrator(config)

    # Collect queries
    if args.query:
        queries = [args.query]
    elif args.queries_file:
        queries = load_queries_from_file(args.queries_file)
        if not args.quiet:
            print(f"  Loaded {len(queries)} queries from {args.queries_file}")
    elif args.benchmark:
        queries = load_benchmark_queries(args.benchmark)
        if not queries:
            print(f"Error: No queries found for benchmark '{args.benchmark}'")
            sys.exit(1)
        if not args.quiet:
            print(f"  Loaded {len(queries)} queries from {args.benchmark} benchmark")
    else:
        print("Error: No query specified")
        sys.exit(1)

    # Execute
    if len(queries) == 1:
        if not args.quiet:
            print(f"\nRunning attack on: \"{queries[0][:80]}\"\n")
        result = run_single_query(orchestrator, queries[0], output_dir, args.no_save)
        print_result(result, verbose=args.verbose)
    else:
        if not args.quiet:
            print(f"\nRunning batch attack on {len(queries)} queries...\n")
        results = run_batch(orchestrator, queries, output_dir, args.no_save)
        print_batch_summary(results, verbose=args.verbose)

    if not args.quiet:
        print("Done.")


if __name__ == "__main__":
    main()