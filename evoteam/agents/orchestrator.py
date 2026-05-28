"""
EvoTeam Orchestrator - Main coordination engine.

Manages the 4-phase serial pipeline:
  Phase 0: Offline fingerprinting + plan pre-generation
  Phase 1: Reconnaissance + plan adaptation
  Phase 2: Tool generation + genetic evolution + self-healing
  Phase 3: Multi-turn attack execution + judge evaluation + cache update

All phases execute sequentially within each iteration.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import time

from ..config import EvoTeamConfig
from ..models.openai_model import LocalModel, LocalModelError
from ..data_structures.attack_plan import AttackPlan
from ..data_structures.attack_tool import AttackTool
from ..data_structures.session_context import SessionContext

from ..agents.fingerprint_agent import FingerprintAgent
from ..agents.plan_generator import PlanGenerator
from ..agents.reconnaissance_agent import ReconnaissanceAgent
from ..agents.tool_synthesizer import ToolSynthesizer
from ..agents.exploitation_agent import ExploitationAgent
from ..agents.judge import LLMJudge

from ..evolution.genetic_algorithm import GeneticOptimizer, ToolPopulation
from ..evolution.diversity_controller import DiversityController
from ..evolution.reflection import LLMReflection

from ..sandbox.sandbox_executor import SandboxExecutor
from ..sandbox.self_healing import SelfHealingEngine

from ..memory.tagged_cache import TaggedCache

from ..utils.query_tagger import QueryTagger, get_nearest_tag
from ..utils.data_saver import DataSaver
from ..utils.logger import EvoTeamLogger


class EvoTeamOrchestrator:
    """Main orchestrator for the EvoTeam attack framework.

    Coordinates all phases in a strict serial pipeline:
    Fingerprinting -> Reconnaissance -> Tool Creation -> Exploitation -> Judge -> Evolve
    """

    def __init__(self, config: EvoTeamConfig):
        self.config = config

        # Generate session ID early for logging
        import uuid
        session_id = str(uuid.uuid4())

        # Create logger (console-only initially, session folder set later)
        self.logger = EvoTeamLogger(config=config, session_id=session_id,
                                    log_dir=config.logs_dir)

        self.logger.sep()
        self.logger.info("orchestrator_init",
            attack_model=config.attack_model,
            target_model=config.target_model,
            judge_model=config.judge_model,
            embed_model=config.embedding_model,
            max_iterations=config.max_iterations,
            population_size=config.population_size)

        attack_base = config.attack_api_base or config.api_base
        target_base = config.target_api_base or config.api_base
        judge_base = config.judge_api_base or config.api_base
        embed_base = config.embed_api_base or config.api_base

        self.attack_model = LocalModel(
            model_name=config.attack_model,
            api_base=attack_base, api_key=config.api_key, temperature=0.7,
            logger=self.logger, raise_on_error=True,
        )
        self.target_model = LocalModel(
            model_name=config.target_model,
            api_base=target_base, api_key=config.api_key, temperature=0.7,
            logger=self.logger,
        )
        self.judge_model = LocalModel(
            model_name=config.judge_model,
            api_base=judge_base, api_key=config.api_key, temperature=0.0,
            logger=self.logger,
        )
        self.embed_model = LocalModel(
            model_name=config.embedding_model,
            api_base=embed_base, api_key=config.api_key, temperature=0.0,
            logger=self.logger,
        )

        # Phase 0 components
        self.fingerprint_agent = FingerprintAgent(self.target_model, self.attack_model)
        self.plan_generator = PlanGenerator(
            llm_model=self.attack_model,
            plan_library_path="./cache/plan_library.json",
        )
        self.query_tagger = QueryTagger()

        # Phase 1 components
        self.recon_agent = ReconnaissanceAgent(self.attack_model)

        # Phase 2 components
        self.sandbox = SandboxExecutor(
            timeout=config.sandbox_timeout,
            allowed_imports=set(config.allowed_imports),
            max_code_length=config.max_code_length,
        )
        self.self_healing = SelfHealingEngine(
            max_retries=config.self_healing_retries,
            sandbox_executor=self.sandbox,
            llm_model=self.attack_model,
        )
        self.tool_synthesizer = ToolSynthesizer(
            attack_model=self.attack_model,
            sandbox=self.sandbox,
            self_healing=self.self_healing,
            max_code_length=config.max_code_length,
        )

        # Phase 3 components
        self.judge = LLMJudge(
            llm_model=self.judge_model,
            success_threshold=config.success_threshold,
        )
        self.exploitation = ExploitationAgent(
            attack_model=self.attack_model,
            target_model=self.target_model,
            judge_model=self.judge_model,
            max_rounds=config.multi_turn_max_rounds,
            stop_score=config.multi_turn_stop_score,
            logger=self.logger,
        )

        # Evolution components
        self.population = ToolPopulation(size=config.population_size)
        self.genetic_optimizer = GeneticOptimizer(
            population=self.population,
            llm_model=self.attack_model,
            tournament_size=config.tournament_size,
            crossover_rate=config.crossover_rate,
            mutation_rate=config.mutation_rate,
            elite_count=config.elite_count,
        )
        self.diversity_controller = DiversityController(
            embedding_model=self.embed_model,
            threshold=config.diversity_threshold,
        )
        self.reflection = LLMReflection(llm_model=self.attack_model)

        # Memory
        self.cache = TaggedCache(
            capacity=config.cache_capacity,
            persistence_path=config.cache_persistence_path,
        )

        # Data saving
        self.data_saver = DataSaver(base_path=config.logs_dir, logger=self.logger)

        # Session tracking
        self.context: Optional[SessionContext] = None
        self.evolution_guidance: str = ""

        # Hall of Fame
        self.hall_of_fame: List[Any] = []
        self.hof_size: int = 2

        self.logger.info("orchestrator_ready")

    def run_attack(self, query: str) -> Dict[str, Any]:
        """Run a complete attack session for a single query.

        Args:
            query: The harmful query to attack with

        Returns:
            Dict with session results
        """
        self.logger.info("session_start", query=query, tag="...")

        # Initialize session
        tag_info = self.query_tagger.tag_with_difficulty(query)
        tag = tag_info["primary_tag"]

        self.context = SessionContext(
            original_query=query,
            query_tag=tag,
            target_model_config={"model_name": self.config.target_model},
            judge_model_config={"model_name": self.config.judge_model},
            attack_model_config={"model_name": self.config.attack_model},
            tagged_cache=self.cache,
        )

        # Create session folder
        session_folder = self.data_saver.create_session_folder(
            query=query,
            target_model=self.config.target_model,
        )

        # === PHASE 0: Fingerprinting + Plan Pre-generation ===
        with self.logger.phase_timer("fingerprinting"):
            self.context.set_phase("fingerprinting")
            self.logger.set_phase("fingerprinting")
            fingerprint = self._run_fingerprinting(query, tag)

        with self.logger.phase_timer("plan_generation"):
            self.context.set_phase("plan_pre_generation")
            self.logger.set_phase("plan_generation")
            plans = self._get_or_generate_plans(fingerprint, tag, query)

        # Save fingerprint details
        self.data_saver.save_fingerprint(fingerprint, session_folder)

        # === Check Cache for Warm Start ===
        cached_tools = self.cache.get_tools(tag, n=3)
        cached_plans = self.cache.get_plans(tag, n=3)
        cached_approaches = self.cache.get_best_approaches(tag)

        if cached_tools or cached_plans:
            self.logger.info("cache_check", tag=tag, cache_hit=True,
                num_tools=len(cached_tools), num_plans=len(cached_plans),
                best_approaches=cached_approaches)
            warm_start = True
        else:
            self.logger.info("cache_check", tag=tag, cache_hit=False)
            warm_start = False

        # === Main Attack Loop ===
        self.logger.info("attack_loop_start", max_iterations=self.config.max_iterations)

        session_results = {
            "query": query,
            "tag": tag,
            "iterations": [],
            "final_success": False,
            "best_score": 0,
            "successful_attacks": [],
        }

        all_reflections = []
        best_overall_score = 0
        tools = []
        consecutive_timeouts = 0

        for iteration in range(1, self.config.max_iterations + 1):
            self.logger.set_iteration(iteration)
            self.logger.info("iteration_start", iteration=iteration,
                             max_iterations=self.config.max_iterations)
            self.context.current_iteration = iteration

            iter_result = {
                "iteration": iteration,
                "phase_results": {},
            }

            try:
                # PHASE 1: Reconnaissance
                with self.logger.phase_timer("reconnaissance", iteration=iteration):
                    self.context.set_phase("reconnaissance")
                    self.logger.set_phase("reconnaissance")
                    intelligence = self._run_reconnaissance(
                        query, tag, fingerprint, plans, warm_start, cached_approaches
                    )
                iter_result["phase_results"]["reconnaissance"] = {
                    "concepts_count": len(intelligence.get("attack_concepts", [])),
                }

                # PHASE 2: Tool Generation & Evolution
                with self.logger.phase_timer("tool_creation", iteration=iteration):
                    self.context.set_phase("tool_creation")
                    self.logger.set_phase("tool_creation")
                    if iteration == 1 and warm_start:
                        tools = self._initialize_population_from_cache(
                            query, tag, cached_tools, intelligence, plans
                        )
                    elif iteration == 1:
                        tools = self._create_initial_tools(
                            query, tag, intelligence, plans
                        )
                    else:
                        tools = self._evolve_tools(query, all_reflections)

                    self.context.tool_population = tools
                iter_result["phase_results"]["tool_creation"] = {
                    "population_size": len(tools),
                    "population_stats": self.population.get_stats() if self.population.tools else {},
                }

                # Log population snapshot for this generation
                if self.config.log_population_snapshots:
                    div_stats = self.diversity_controller.get_diversity_stats(tools)
                    self.logger.log_population_snapshot(
                        generation=self.population.generation, tools=tools,
                        diversity_stats=div_stats)
                    self.data_saver.save_population_snapshot(
                        generation=self.population.generation, tools=tools,
                        diversity_stats=div_stats, folder=session_folder)

                # PHASE 3: Exploitation & Judging
                with self.logger.phase_timer("exploitation", iteration=iteration):
                    self.context.set_phase("exploitation")
                    self.logger.set_phase("exploitation")
                    exploitation_results = self._run_exploitation(
                        query, tools, plans, intelligence
                    )
                iter_result["phase_results"]["exploitation"] = {
                    "attacks_run": len(exploitation_results),
                    "best_turn_score": max(
                        [r.get("highest_score", 0) for r in exploitation_results], default=0
                    ),
                }

                # PHASE 3b: LLM Reflection
                with self.logger.phase_timer("reflection", iteration=iteration):
                    self.context.set_phase("reflection")
                    self.logger.set_phase("reflection")
                    reflections = self._run_reflection(
                        exploitation_results, tools, query, intelligence
                    )
                    all_reflections.extend(reflections)

                # Update tool performance based on exploitation results
                self._update_tool_performance(tools, exploitation_results)

                # Apply diversity control
                self.diversity_controller.apply_diversity_penalties(tools)

                # Update Hall of Fame — preserve best tools permanently
                self._update_hall_of_fame(tools)

                # Track best score
                for r in exploitation_results:
                    score = r.get("highest_score", 0)
                    if score > best_overall_score:
                        best_overall_score = score
                    if r.get("conversation_successful"):
                        session_results["successful_attacks"].append({
                            "tool_name": r.get("tool_name", "Unknown"),
                            "highest_score": score,
                            "turns": r.get("total_turns", 0),
                        })

                iter_result["best_score_this_iteration"] = max(
                    [r.get("highest_score", 0) for r in exploitation_results], default=0
                )
                session_results["iterations"].append(iter_result)

                self.logger.info("iteration_end", iteration=iteration,
                    best_score=iter_result["best_score_this_iteration"],
                    population_avg_fitness=round(self.population.get_stats().get("avg_fitness", 0), 3))

                # Generate evolution guidance
                if all_reflections:
                    self.evolution_guidance = self.reflection.generate_evolution_guidance(
                        reflections=all_reflections[-3:],
                        population_stats=self.population.get_stats(),
                    )

                # Check for success
                if best_overall_score >= self.config.success_threshold:
                    self.logger.info("attack_succeeded",
                        score=best_overall_score, threshold=self.config.success_threshold)
                    session_results["final_success"] = True
                    break

            except LocalModelError as e:
                consecutive_timeouts += 1
                self.logger.error("iteration_failed", iteration=iteration,
                    error=str(e), model=e.model_name, timeout_count=consecutive_timeouts)
                if consecutive_timeouts >= 3:
                    self.logger.error("model_unreachable_skipping_query",
                        model=e.model_name, consecutive_failures=consecutive_timeouts)
                    break
                continue

        # === Finalize ===
        session_results["best_score"] = best_overall_score
        session_results["final_success"] = best_overall_score >= self.config.success_threshold
        session_results["total_iterations"] = iteration
        session_results["population_final_stats"] = self.population.get_stats()

        # Update cache with successful results
        self._update_cache(query, tag, session_results)

        # Save results — all conversations (NEW) + legacy successful-only
        self.logger.info("saving_results")
        self.data_saver.save_session_results(session_results, session_folder)
        self.data_saver.save_tool_population(tools, session_folder)
        self.data_saver.save_attack_history(self.context, session_folder)
        if self.config.log_all_conversations:
            self.data_saver.save_all_conversations(self.context, session_folder)
        self.data_saver.save_reflections(all_reflections, session_folder)
        self.data_saver.save_summary(self.context, session_folder)

        self.logger.info("session_end",
            query=query, tag=tag, iterations=iteration,
            best_score=best_overall_score, final_success=session_results['final_success'],
            tools_created=len(self.context.created_tools))
        self.logger.sep("#", 50)
        self.logger.close()

        return session_results

    def run_attack_batch(self, queries: List[str]) -> List[Dict]:
        """Run attacks against multiple queries sequentially.

        Cross-query learning via tagged cache: later queries benefit from
        tools and plans cached by earlier queries with the same tag.
        """
        results = []
        for i, query in enumerate(queries, 1):
            self.logger.info("batch_query", index=i, total=len(queries), query=query[:80])
            result = self.run_attack(query)
            results.append(result)
        return results

    # ========== Phase 0 Methods ==========

    def _run_fingerprinting(self, query: str, tag: str) -> Dict:
        """Run model fingerprinting."""
        self.logger.debug("fingerprinting_start")
        fingerprint = self.fingerprint_agent.probe()
        self.context.model_fingerprint = fingerprint

        self.logger.info("fingerprint_complete",
            refusal_rate=fingerprint.get('refusal_rate', 0),
            recommended_approaches=fingerprint.get('recommended_approaches', []))
        return fingerprint

    def _get_or_generate_plans(
        self, fingerprint: Dict, tag: str, query: str
    ) -> List[AttackPlan]:
        """Get cached plans or generate new ones."""
        plans = self.plan_generator.get_or_generate(
            fingerprint=fingerprint,
            tag=tag,
            n=3,
        )
        self.logger.debug("plans_available", tag=tag, count=len(plans))
        return plans

    # ========== Phase 1 Methods ==========

    def _run_reconnaissance(
        self,
        query: str,
        tag: str,
        fingerprint: Dict,
        plans: List[AttackPlan],
        warm_start: bool,
        cached_approaches: List[str],
    ) -> Dict:
        """Run reconnaissance to generate attack intelligence."""
        cache_info = None
        if warm_start:
            cache_info = {
                "tag": tag,
                "avg_score": self.cache.get_stats(tag).get("avg_score", 0),
                "sample_count": self.cache.get_stats(tag).get("sample_count", 0),
                "best_approaches": cached_approaches,
            }

        intelligence = self.recon_agent.gather_intelligence(
            query=query,
            query_tag=tag,
            fingerprint=fingerprint,
            matched_plans=plans,
            cache_info=cache_info,
        )

        # Log raw LLM output when recon fails to extract concepts
        if len(intelligence.get("attack_concepts", [])) == 0:
            self.logger.warning("recon_empty_concepts",
                full_response=intelligence.get("full_response", "")[:2000])

        self.logger.info("reconnaissance_complete",
            concepts_count=len(intelligence.get("attack_concepts", [])))
        return intelligence

    # ========== Phase 2 Methods ==========

    def _create_initial_tools(
        self,
        query: str,
        tag: str,
        intelligence: Dict,
        plans: List[AttackPlan],
    ) -> List[AttackTool]:
        """Create initial tool population from reconnaissance intelligence."""
        self.logger.debug("creating_initial_population")
        tools = []

        # Create one tool per plan (or at least 3 tools)
        plan_list = plans[:3] if plans else [None, None, None]

        for i, plan in enumerate(plan_list):
            tool = self.tool_synthesizer.create_tool(
                query=query,
                query_tag=tag,
                intelligence=intelligence,
                plan=plan,
            )
            if tool:
                tool.generation = 0
                tools.append(tool)
                self.context.add_tool(tool)
                self.logger.debug("tool_created", tool_name=tool.tool_name, generation=0)

        # Fill to population size with additional variants
        attempts = 0
        while len(tools) < self.config.population_size and attempts < 15:
            tool = self.tool_synthesizer.create_tool(
                query=query, query_tag=tag, intelligence=intelligence, plan=None)
            if tool:
                tool.generation = 0
                tools.append(tool)
                self.context.add_tool(tool)
                self.logger.debug("tool_created", tool_name=tool.tool_name, generation=0)
            attempts += 1

        self.population.initialize(tools)
        self.logger.info("population_initialized", size=len(tools), warm_start=False)
        return tools

    def _initialize_population_from_cache(
        self,
        query: str,
        tag: str,
        cached_tools: List[Dict],
        intelligence: Dict,
        plans: List[AttackPlan],
    ) -> List[AttackTool]:
        """Warm-start population using cached tools."""
        self.logger.debug("warm_start_from_cache")
        tools = []

        # Reconstruct tools from cache
        for tool_dict in cached_tools:
            try:
                tool = AttackTool.from_dict(tool_dict)
                tool.generation = 0
                tools.append(tool)
                self.context.add_tool(tool)
                self.logger.debug("cached_tool_loaded", tool_name=tool.tool_name,
                    score=tool.performance.avg_judge_score)
            except Exception as e:
                self.logger.warning("cached_tool_load_failed", error=str(e))

        # Fill remaining slots with new tools
        while len(tools) < self.config.population_size:
            tool = self.tool_synthesizer.create_tool(
                query=query,
                query_tag=tag,
                intelligence=intelligence,
                plan=plans[len(tools) % len(plans)] if plans else None,
            )
            if tool:
                tool.generation = 0
                tools.append(tool)
                self.context.add_tool(tool)

        self.population.initialize(tools)
        self.logger.info("population_initialized", size=len(tools), warm_start=True)
        return tools

    def _evolve_tools(
        self,
        query: str,
        reflections: List[Dict],
    ) -> List[AttackTool]:
        """Evolve the tool population using genetic algorithm."""
        self.logger.debug("evolving_population", generation=self.population.generation + 1)

        context = {
            "query": query,
            "evolution_guidance": self.evolution_guidance[:500] if self.evolution_guidance else "",
            "reflections": [r.get("reflection_text", "")[:500] for r in reflections[-3:]],
        }

        new_population = self.genetic_optimizer.evolve_one_generation(query, context)

        # Inject Hall of Fame — replace worst tools with preserved bests
        if self.hall_of_fame:
            new_population.sort(key=lambda t: t.fitness)
            inject_count = min(len(self.hall_of_fame), len(new_population) // 3)
            for i in range(inject_count):
                new_population[i] = self.hall_of_fame[i]
            self.logger.debug("hall_of_fame_injected", count=inject_count)

        for tool in new_population:
            if tool not in self.context.created_tools:
                self.context.add_tool(tool)

        stats = self.population.get_stats()
        self.logger.info("population_evolved", generation=self.population.generation,
            size=len(new_population),
            avg_fitness=round(stats.get("avg_fitness", 0), 3),
            max_fitness=round(stats.get("max_fitness", 0), 3))

        return new_population

    # ========== Phase 3 Methods ==========

    def _run_exploitation(
        self,
        query: str,
        tools: List[AttackTool],
        plans: List[AttackPlan],
        intelligence: Dict,
    ) -> List[Dict]:
        """Run multi-turn attacks using all tools in population."""
        self.logger.info("exploitation_start", tool_count=len(tools))

        # Build execution context for tools
        exec_context = {
            "target_model": self.target_model,
            "judge_model": self.judge_model,
            "attack_model": self.attack_model,
            "configuration": self.config.to_dict(),
        }

        results = []
        for i, tool in enumerate(tools):
            self.logger.debug("tool_testing", index=i+1, total=len(tools),
                tool_name=tool.tool_name)

            # Select best plan for this tool
            plan = plans[i % len(plans)] if plans else None

            # Adapt plan if available
            if plan:
                plan = self.plan_generator.adapt_plan(
                    plan=plan,
                    query=query,
                    attack_history=self.context.attack_history[-5:],
                )

            # Test tool execution
            start_time = time.time()
            tool_test = self.tool_synthesizer.test_tool(
                tool=tool,
                query=query,
                context=exec_context,
            )
            exec_time = time.time() - start_time

            # Execute multi-turn attack using the tool's output
            attack_result = self.exploitation.execute_tool_based_attack(
                query=query,
                tool=tool,
                context={"sandbox_executor": self.sandbox.execute},
                plan=plan,
            )

            attack_result["tool_name"] = tool.tool_name
            attack_result["tool_id"] = tool.tool_id
            attack_result["execution_time"] = exec_time

            # Record execution in tool
            tool.record_execution(
                success=tool_test.get("success", False),
                execution_time=exec_time,
                judge_score=attack_result.get("highest_score", 1),
                query=query,
                self_healing_used=tool_test.get("healing_attempts", 0) > 0,
                self_healing_succeeded=tool_test.get("healing_successful", False),
            )

            # Record attack in context
            self.context.add_attack_result({
                "original_query": query,
                "tool_name": tool.tool_name,
                "tool_id": tool.tool_id,
                "final_judge_score": attack_result.get("highest_score", 1),
                "attack_successful": attack_result.get("conversation_successful", False),
                "total_turns": attack_result.get("total_turns", 0),
                "multi_turn_results": attack_result,
                "timestamp": datetime.now().isoformat(),
            })

            results.append(attack_result)

            if attack_result.get("conversation_successful"):
                self.logger.info("tool_attack_success",
                    tool_name=tool.tool_name,
                    score=attack_result['highest_score'],
                    turns=attack_result['total_turns'])

        return results

    def _run_reflection(
        self,
        exploitation_results: List[Dict],
        tools: List[AttackTool],
        query: str,
        intelligence: Dict,
    ) -> List[Dict]:
        """Run LLM reflection on exploitation results."""
        self.logger.debug("reflection_start", attack_count=len(exploitation_results))
        reflections = []

        # Reflect on best and worst tools
        tools_by_score = sorted(
            zip(tools, exploitation_results),
            key=lambda x: x[1].get("highest_score", 0),
            reverse=True,
        )

        # Reflect on top 2 and bottom 1
        targets = tools_by_score[:2] + tools_by_score[-1:]

        for tool, result in targets:
            reflection = self.reflection.analyze_attack(
                tool_name=tool.tool_name,
                tool_code=tool.tool_code,
                tool_description=tool.tool_description,
                query=query,
                conversation_history=result.get("conversation_history", []),
                judge_score=result.get("highest_score", 1),
                attack_successful=result.get("conversation_successful", False),
            )
            reflections.append(reflection)

        return reflections

    def _update_hall_of_fame(self, tools: List[Any]):
        """Preserve top-performing tools permanently, immune to GA replacement."""
        for tool in tools:
            if tool.performance.avg_judge_score >= 3.0:  # Partial compliance or better
                existing = [t for t in self.hall_of_fame if t.tool_name == tool.tool_name]
                if not existing:
                    self.hall_of_fame.append(tool)
        # Keep only top-N by avg_judge_score
        self.hall_of_fame.sort(
            key=lambda t: t.performance.avg_judge_score, reverse=True
        )
        self.hall_of_fame = self.hall_of_fame[:self.hof_size]
        if self.hall_of_fame:
            scores = [f"{t.tool_name}({t.performance.avg_judge_score:.1f})" for t in self.hall_of_fame]
            self.logger.debug("hall_of_fame_updated", tools=str(scores))

    def _update_tool_performance(
        self, tools: List[AttackTool], results: List[Dict]
    ):
        """Update tool performance based on exploitation results."""
        for tool, result in zip(tools, results):
            tool.diversity_penalty = 0.0  # Reset before recalculation

    # ========== Cache Methods ==========

    def _update_cache(self, query: str, tag: str, session_results: Dict):
        """Update tagged cache with successful results."""
        successful = session_results.get("successful_attacks", [])
        if not successful:
            self.logger.debug("cache_update_skipped", tag=tag, reason="no_successful_attacks")
            return

        # Cache successful tools
        for success in successful:
            tool_name = success.get("tool_name", "")
            for tool in self.context.tool_population:
                if tool.tool_name == tool_name and tool.performance.avg_judge_score >= 4:
                    self.cache.add_tool(tag, tool)
                    break

        # Cache successful plans (use correct fingerprint hash)
        fp_hash = self.plan_generator._hash_fingerprint(
            self.context.model_fingerprint
        )
        for plan in self.plan_generator.plan_library.get(fp_hash, {}).get(tag, []):
            if plan.success_rate > 0.5:
                self.cache.add_plan(tag, plan)

        # Track best approaches
        approaches = set()
        for tool in self.context.tool_population:
            if tool.performance.avg_judge_score >= 4:
                approach = tool.approach or tool.tool_category
                if approach:
                    approaches.add(approach)
        for approach in approaches:
            self.cache.add_approach(tag, approach)

        self.logger.info("cache_updated", tag=tag,
            tools_count=len(self.cache.get_tools(tag)),
            plans_count=len(self.cache.get_plans(tag)))