from __future__ import annotations

from typing import Any, Optional
import json
import re
import numpy as np
import logging

from agents.agent_base import BaseAgent
from agents.schemas import ReasonerChannelDecision, ReasonerInput, ReasonerOutput
from core.registry import AGENT_REGISTRY
from llm.client_base import LLMRequest
from tasks.prompt_targets import get_prompt_target
from prompts.templates import reasoner as reasoner_templates

logger = logging.getLogger(__name__)


@AGENT_REGISTRY.decorator("reasoner_agent")
class ReasonerAgent(BaseAgent):
	"""
	Reasoning agent that makes decisions based on retrieved results.

	For each channel, analyzes retrieved neighbors and produces a prediction,
	confidence score, and optional reasoning.
	"""

	def validate_input(self, input_data: Any) -> None:
		if not isinstance(input_data, ReasonerInput):
			raise TypeError(
				f"{self.name}: input_data must be ReasonerInput, "
				f"but got {type(input_data).__name__}."
			)

	def _run_impl(
		self,
		input_data: ReasonerInput,
		context: Optional[dict[str, Any]] = None,
	) -> ReasonerOutput:
		query = input_data.query
		task_spec = input_data.task_spec
		retrieved_sets = input_data.retrieved_sets

		channel_decisions: list[ReasonerChannelDecision] = []
		decision_method = str(self.get_config("decision_method", "majority_vote"))

		for channel_id, retrieved_set in retrieved_sets.items():
			decision = self._make_channel_decision(
				channel_id=channel_id,
				retrieved_set=retrieved_set,
				query=query,
				task_spec=task_spec,
				method=decision_method,
				context=context,
			)
			channel_decisions.append(decision)

		self.log_info(
			context,
			"ReasonerAgent '%s': made decisions for %d channels",
			self.name,
			len(channel_decisions),
		)

		return ReasonerOutput(
			query_id=query.query_id,
			channel_decisions=channel_decisions,
			metadata={
				"sample_id": query.sample.sample_id,
				"num_channels": len(channel_decisions),
				"decision_method": decision_method,
				"task_type": task_spec.task_type.value,
			},
		)

	def _make_channel_decision(
		self,
		channel_id: int,
		retrieved_set: Any,
		query: Any,
		task_spec: Any,
		method: str,
		context: Optional[dict[str, Any]] = None,
	) -> ReasonerChannelDecision:
		"""Make a decision for a single channel based on retrieved neighbors."""
		if not retrieved_set.examples:
			return ReasonerChannelDecision(
				channel_id=channel_id,
				prediction=None,
				confidence=0.0,
				reasoning="No neighbors retrieved",
				metadata={"num_neighbors": 0},
			)

		use_llm = bool(self.get_config("use_llm", False))
		# Debug: use logger.debug so output is controllable
		logger.debug("ReasonerAgent._make_channel_decision: channel_id=%s use_llm=%s agent_name=%s", channel_id, use_llm, self.name)
		if use_llm:
			return self._llm_reasoning_decision(
				channel_id=channel_id,
				retrieved_set=retrieved_set,
				query=query,
				task_spec=task_spec,
				context=context,
			)

		if method == "majority_vote":
			return self._majority_vote_decision(channel_id, retrieved_set)
		elif method == "weighted_vote":
			return self._weighted_vote_decision(channel_id, retrieved_set)
		elif method == "top_1":
			return self._top_1_decision(channel_id, retrieved_set)
		else:
			return self._majority_vote_decision(channel_id, retrieved_set)


	def _majority_vote_decision(self, channel_id: int, retrieved_set: Any) -> ReasonerChannelDecision:
		"""Make decision using simple majority vote (helper for LLM fallback)."""
		from collections import Counter

		labels = retrieved_set.labels
		label_counts = Counter(labels)
		most_common_label = label_counts.most_common(1)[0][0] if label_counts else None

		if most_common_label is None:
			return ReasonerChannelDecision(
				channel_id=channel_id,
				prediction=None,
				confidence=0.0,
				reasoning="All neighbors have None label",
			)

		max_count = label_counts[most_common_label]
		total_count = len(labels)
		confidence = float(max_count) / float(total_count) if total_count > 0 else 0.0

		reasoning = f"Majority vote: {most_common_label} ({max_count}/{total_count} neighbors)"

		return ReasonerChannelDecision(
			channel_id=channel_id,
			prediction=most_common_label,
			confidence=confidence,
			reasoning=reasoning,
			metadata={
				"num_neighbors": len(labels),
				"vote_counts": dict(label_counts),
			},
		)

	def _weighted_vote_decision(self, channel_id: int, retrieved_set: Any) -> ReasonerChannelDecision:
		"""Make decision using weighted voting based on retrieval scores."""
		label_scores: dict[Any, float] = {}

		for example in retrieved_set.examples:
			label = example.label
			score = float(example.score.value)

			if example.score.higher_is_better:
				weight = score
			else:
				weight = 1.0 / (1.0 + score)

			if label not in label_scores:
				label_scores[label] = 0.0
			label_scores[label] += weight

		if not label_scores:
			return ReasonerChannelDecision(
				channel_id=channel_id,
				prediction=None,
				confidence=0.0,
				reasoning="No valid labels in neighbors",
			)

		best_label = max(label_scores.keys(), key=lambda l: label_scores[l])
		total_weight = sum(label_scores.values())
		confidence = float(label_scores[best_label]) / float(total_weight) if total_weight > 0 else 0.0

		reasoning = f"Weighted vote: {best_label} (weight={label_scores[best_label]:.3f})"

		return ReasonerChannelDecision(
			channel_id=channel_id,
			prediction=best_label,
			confidence=confidence,
			reasoning=reasoning,
			metadata={
				"num_neighbors": len(retrieved_set.examples),
				"label_scores": {str(k): float(v) for k, v in label_scores.items()},
			},
		)

	def _top_1_decision(self, channel_id: int, retrieved_set: Any) -> ReasonerChannelDecision:
		"""Make decision based on top-1 neighbor."""
		if not retrieved_set.examples:
			return ReasonerChannelDecision(
				channel_id=channel_id,
				prediction=None,
				confidence=0.0,
				reasoning="No neighbors retrieved",
			)

		top_example = retrieved_set.examples[0]
		confidence_val = 1.0 / (1.0 + float(top_example.score.value))

		reasoning = f"Top-1 neighbor: {top_example.sample_id} (label={top_example.label})"

		return ReasonerChannelDecision(
			channel_id=channel_id,
			prediction=top_example.label,
			confidence=float(confidence_val),
			reasoning=reasoning,
			metadata={
				"top_1_sample_id": top_example.sample_id,
				"top_1_score": float(top_example.score.value),
			},
		)

	def _llm_reasoning_decision(self, channel_id: int, retrieved_set: Any, query: Any, task_spec: Any, context: Optional[dict[str, Any]] = None) -> ReasonerChannelDecision:
		"""Use LLM to reason about neighbors and the query for one channel.
		Falls back to majority vote on any error or missing client.
		"""
		llm_client = (context or {}).get("llm_client") if context is not None else None
		llm_trace: dict[str, Any] = {
			"llm_attempted": llm_client is not None,
			"llm_used": False,
			"llm_raw_response": None,
			"llm_parsed": None,
			"llm_error": None,
		}
		if llm_client is None:
			majority = self._majority_vote_decision(channel_id, retrieved_set)
			meta = dict(majority.metadata or {})
			meta.update(llm_trace)
			meta["llm_error"] = "llm_client_missing"
			return ReasonerChannelDecision(
				channel_id=channel_id,
				prediction=majority.prediction,
				confidence=majority.confidence,
				reasoning=majority.reasoning,
				metadata=meta,
			)

		try:
			# Diagnostic: record whether llm_client is present
			self.log_info(context, "_llm_reasoning_decision: llm_client_present=%s", llm_client is not None)
			self.log_event(context, event_type="llm_attempt_start", payload={"channel_id": channel_id, "num_neighbors": len(retrieved_set.examples)})
			# extract representations
			query_raw = self._extract_channel_series(query.sample, channel_id)
			query_text = self._compute_text_summary(query_raw)
			query_stats = self._compute_statistics(query_raw)

			# Prefer runner-provided per-channel statistics when available.
			if context is not None:
				by_channel = context.get("query_stat_by_channel")
				if isinstance(by_channel, dict) and channel_id in by_channel:
					payload = by_channel[channel_id]
					if isinstance(payload, dict):
						query_stats = payload
					else:
						try:
							query_stats = dict(payload)
						except Exception:
							pass

			prompt = self._build_llm_prompt(
				channel_id=channel_id,
				query_raw=query_raw,
				query_text=query_text,
				query_stats=query_stats,
				retrieved_set=retrieved_set,
				task_spec=task_spec,
			)
			llm_trace["llm_prompt_len"] = len(prompt)
			llm_trace["llm_prompt"] = prompt
			llm_trace["llm_prompt_preview"] = prompt[:4000]
			if isinstance(query_stats, dict):
				llm_trace["query_stat_keys"] = list(query_stats.keys())

			# Send to LLM (wrapped by retry in client)
			try:
				req = LLMRequest(messages=[{"role": "user", "content": prompt}], system=reasoner_templates.SYSTEM_PROMPT)
				logger.debug("llm prompt len=%d", len(prompt))
				resp = llm_client.complete(req)
				text = getattr(resp, "content", str(resp))
				logger.debug("llm resp repr: %s", repr(resp))
				self.log_info(context, "_llm_reasoning_decision: llm_raw_response_len=%d", len(text) if isinstance(text, str) else 0)
				self.log_event(context, event_type="llm_raw_response", payload={"len": len(text) if isinstance(text, str) else 0})
				if isinstance(text, str):
					logger.debug("llm raw response: %s", (text[:2000] + '...') if len(text) > 2000 else text)
				else:
					logger.debug("llm raw response: non-string response; repr: %s", repr(text))
				try:
					parsed = self._extract_json_from_response(text)
					logger.debug("llm parsed repr: %s", repr(parsed))
					llm_trace["llm_parsed"] = parsed
				except Exception as parse_e:
					logger.exception("llm parse error: %s", str(parse_e))
					logger.debug("llm raw for parse error: %s", (text[:2000] + '...') if isinstance(text, str) and len(text) > 2000 else text)
					llm_trace["llm_raw_response"] = text if isinstance(text, str) else repr(text)
					llm_trace["llm_error"] = str(parse_e)
					raise
			except Exception as llme:
				self.log_info(context, "_llm_reasoning_decision: llm call raised %s", str(llme))
				self.log_event(context, event_type="llm_call_error", payload={"error": str(llme)})
				llm_trace["llm_error"] = str(llme)
				raise

			pred = parsed.get("label")
			conf = float(parsed.get("confidence", 0.0))
			conf = max(0.0, min(1.0, conf))
			reasoning = str(parsed.get("reasoning", ""))
			llm_trace["llm_raw_response"] = text
			llm_trace["llm_used"] = True

			if pred is None:
				self.log_info(context, "_llm_reasoning_decision: parsed JSON missing 'label', parsed=%s", parsed)
				self.log_event(context, event_type="llm_parsed_no_label", payload={"parsed": parsed})
				majority = self._majority_vote_decision(channel_id, retrieved_set)
				meta = dict(majority.metadata or {})
				meta.update(llm_trace)
				return ReasonerChannelDecision(
					channel_id=channel_id,
					prediction=majority.prediction,
					confidence=majority.confidence,
					reasoning=majority.reasoning,
					metadata=meta,
				)

			self.log_info(context, "_llm_reasoning_decision: parsed_keys=%s", list(parsed.keys()) if isinstance(parsed, dict) else None)
			self.log_event(context, event_type="llm_parsed", payload={"keys": list(parsed.keys()) if isinstance(parsed, dict) else None})

			return ReasonerChannelDecision(
				channel_id=channel_id,
				prediction=pred,
				confidence=conf,
				reasoning=reasoning,
				metadata={
					"num_neighbors": len(retrieved_set.examples),
					**llm_trace,
				},
			)
		except Exception as e:
			majority = self._majority_vote_decision(channel_id, retrieved_set)
			meta = dict(majority.metadata or {})
			meta.update(llm_trace)
			meta.update({"llm_fallback": True, "llm_error": str(e)})
			return ReasonerChannelDecision(
				channel_id=channel_id,
				prediction=majority.prediction,
				confidence=majority.confidence,
				reasoning=majority.reasoning,
				metadata=meta,
			)


	def _extract_channel_series(self, sample: Any, channel_id: int) -> list[float]:
		# Supports common sample layouts: channels/data/values/x.
		try:
			series = None
			if hasattr(sample, "channels"):
				series = sample.channels[channel_id]
			elif hasattr(sample, "data"):
				series = sample.data[channel_id]
			elif hasattr(sample, "x"):
				x = getattr(sample, "x")
				if getattr(x, "ndim", 1) == 1:
					series = x
				elif x.ndim == 2:
					# TimeSeriesSample uses shape (T, C); select the requested channel column.
					series = x[:, channel_id]
			else:
				series = sample.values[channel_id]
			return [float(x) for x in series]
		except Exception:
			return []


	def _compute_text_summary(self, series: list[float]) -> str:
		if not series:
			return ""
		arr = np.array(series)
		return f"mean={arr.mean():.4f}, std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}, len={arr.size}"


	def _compute_statistics(self, series: list[float]) -> dict:
		if not series:
			return {}
		arr = np.array(series)
		return {
			"mean": float(arr.mean()),
			"std": float(arr.std()),
			"min": float(arr.min()),
			"max": float(arr.max()),
			"median": float(np.median(arr)),
			"length": int(arr.size),
		}


	def _extract_json_from_response(self, text: str) -> dict[str, Any]:
		"""Extract and parse JSON from LLM response text.

		Handles cases where LLM returns JSON wrapped in markdown or extra text.
		Falls back to empty dict on parse failure.
		"""
		if not text:
			return {}

		# Try to find JSON block in markdown code fence first.
		json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
		if json_match:
			try:
				return json.loads(json_match.group(1))
			except json.JSONDecodeError:
				pass

		# Then try to parse the last standalone JSON object in the text.
		# This is more robust than a greedy `{.*}` match because the LLM may
		# mention `{}` in its reasoning before the final answer JSON.
		candidate_matches = list(re.finditer(r'\{.*?\}', text, re.DOTALL))
		for match in reversed(candidate_matches):
			candidate = match.group(0)
			try:
				return json.loads(candidate)
			except json.JSONDecodeError:
				continue

		# If all else fails, return empty dict (will trigger fallback)
		return {}


	def _format_series(self, series: list[float], max_len: int = 32) -> str:
		if not series:
			return "[]"
		if len(series) <= max_len:
			return str([float(x) for x in series])
		# show head/tail
		head = [float(x) for x in series[: max_len // 2]]
		tail = [float(x) for x in series[-(max_len // 2) :]]
		return str(head)[:-1] + ", ..., " + str(tail)[1:]


	def _build_llm_prompt(self, channel_id: int, query_raw: list[float], query_text: str, query_stats: dict, retrieved_set: Any, task_spec: Any) -> str:
		"""Build LLM prompt using template with 3 representations and neighbors."""
		# Format query representations
		query_raw_str = self._format_series(query_raw, max_len=64)
		query_stats_str = json.dumps(query_stats)

		# Format neighbors
		neighbors_lines = []
		for ex in retrieved_set.examples[:5]:
			sample_id = getattr(ex, 'sample_id', getattr(ex, 'id', 'unknown'))
			label = getattr(ex, 'label', None)
			score = getattr(ex, 'score', None)

			# Prefer stored statistic payloads; fall back to raw series only if needed.
			nb_stats = self._extract_neighbor_statistics(ex)

			neighbor_text = reasoner_templates.NEIGHBOR_TEMPLATE.format(
				sample_id=sample_id,
				label=label,
				score=score,
				stats=json.dumps(nb_stats),
			)
			neighbors_lines.append(neighbor_text)

		neighbors_block = "\n".join(neighbors_lines) if neighbors_lines else "(no neighbors retrieved)"

		# Extract task info
		task_name = getattr(task_spec, "name", str(getattr(task_spec, 'task_type', 'unknown')))
		task_type = getattr(getattr(task_spec, "task_type", None), "value", str(getattr(task_spec, "task_type", "unknown")))
		task_target = get_prompt_target(getattr(task_spec, "task_type", "unknown"), getattr(task_spec, "label_space", None))
		label_space = getattr(task_spec, "label_space", "(not specified)")
		if isinstance(label_space, (list, tuple)):
			label_space = ", ".join(str(x) for x in label_space)

		# Fill template
		user_prompt = reasoner_templates.USER_TEMPLATE.format(
			channel_id=channel_id,
			query_raw=query_raw_str,
			query_text=query_text,
			query_stats=query_stats_str,
			neighbors_block=neighbors_block,
			task_type=task_type,
			task_target=task_target,
			task_name=task_name,
			label_space=label_space,
		)

		return user_prompt

	def _extract_neighbor_statistics(self, example: Any) -> dict[str, Any]:
		"""Extract a neighbor's statistical payload with feature names when available."""
		for attr_name in ("payload", "statistic_view", "stat_vector"):
			payload = getattr(example, attr_name, None)
			if payload is None:
				continue
			if isinstance(payload, dict):
				return payload
			try:
				return dict(payload)
			except Exception:
				pass

		metadata = getattr(example, "metadata", None)
		if isinstance(metadata, dict):
			for key in ("statistic_view", "stat_vector", "payload"):
				payload = metadata.get(key)
				if payload is None:
					continue
				if isinstance(payload, dict):
					return payload
				try:
					return dict(payload)
				except Exception:
					pass

		for attr_name in ("values", "data", "x"):
			raw = getattr(example, attr_name, None)
			if raw is None:
				continue
			try:
				return self._compute_statistics(raw)
			except Exception:
				continue

		return {}
