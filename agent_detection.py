"""
Agent detection module for fingerprinting AI coding assistants.

Detects agents based on their system prompt patterns. Known agents have
specific identifiers, with fallback to generic "You are X" pattern matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class DetectionResult:
    """Result of agent detection."""

    agent: str  # Detected agent name or "Unknown"
    confidence: str  # "high", "medium", "low"


@dataclass
class AgentPattern:
    """Pattern definition for a known agent."""

    name: str
    patterns: list[re.Pattern[str]]


# Known agent patterns - ordered by specificity
# These patterns are based on documented/leaked system prompts
KNOWN_AGENTS: list[AgentPattern] = [
    # Cline: "You are Cline, a highly skilled software engineer..."
    AgentPattern(
        name="Cline",
        patterns=[re.compile(r"You are Cline\b", re.IGNORECASE)],
    ),
    # Droid (Factory): "You are Droid, an AI software engineering agent..."
    AgentPattern(
        name="Droid",
        patterns=[re.compile(r"You are Droid\b", re.IGNORECASE)],
    ),
    # Cascade (Windsurf): "You are Cascade, a powerful agentic AI coding assistant..."
    AgentPattern(
        name="Cascade",
        patterns=[re.compile(r"You are Cascade\b", re.IGNORECASE)],
    ),
    # Claude Code: "You are Claude Code" (dynamic prompt assembly)
    AgentPattern(
        name="Claude Code",
        patterns=[re.compile(r"You are Claude Code\b", re.IGNORECASE)],
    ),
    # Cursor: "You operate exclusively in Cursor" or "You operate in Cursor"
    AgentPattern(
        name="Cursor",
        patterns=[
            re.compile(r"You operate(?:\s+exclusively)?\s+in Cursor\b", re.IGNORECASE),
            re.compile(r"world's best IDE", re.IGNORECASE),  # Cursor's distinctive claim
        ],
    ),
    # Aider: "You are an expert software developer"
    AgentPattern(
        name="Aider",
        patterns=[re.compile(r"You are aider\b", re.IGNORECASE)],
    ),
    # Continue: "You are a helpful assistant"... with Continue-specific tool patterns
    AgentPattern(
        name="Continue",
        patterns=[re.compile(r"\bContinue\.dev\b", re.IGNORECASE)],
    ),
    # Copilot: GitHub Copilot patterns
    AgentPattern(
        name="Copilot",
        patterns=[
            re.compile(r"GitHub Copilot\b", re.IGNORECASE),
            re.compile(r"You are Copilot\b", re.IGNORECASE),
        ],
    ),
    # Devin: Cognition's autonomous agent
    AgentPattern(
        name="Devin",
        patterns=[re.compile(r"You are Devin\b", re.IGNORECASE)],
    ),
    # OpenHands (formerly OpenDevin)
    AgentPattern(
        name="OpenHands",
        patterns=[
            re.compile(r"You are OpenHands\b", re.IGNORECASE),
            re.compile(r"You are OpenDevin\b", re.IGNORECASE),
        ],
    ),
    # OpenCode
    AgentPattern(
        name="OpenCode",
        patterns=[re.compile(r"You are opencode\b", re.IGNORECASE)],
    ),
    # SWE-agent
    AgentPattern(
        name="SWE-agent",
        patterns=[re.compile(r"SWE-agent\b", re.IGNORECASE)],
    ),
    # Amazon Q Developer
    AgentPattern(
        name="Amazon Q",
        patterns=[re.compile(r"Amazon Q\b", re.IGNORECASE)],
    ),
    # Bolt.new
    AgentPattern(
        name="Bolt",
        patterns=[re.compile(r"You are Bolt\b", re.IGNORECASE)],
    ),
    # Lovable (formerly Loveable)
    AgentPattern(
        name="Lovable",
        patterns=[re.compile(r"You are Lovable\b", re.IGNORECASE)],
    ),
    # Replit Agent
    AgentPattern(
        name="Replit",
        patterns=[
            re.compile(r"You are Replit\b", re.IGNORECASE),
            re.compile(r"Replit Agent\b", re.IGNORECASE),
        ],
    ),
]

# Generic fallback pattern: "You are X" where X is a single word (more restrictive)
GENERIC_YOU_ARE_PATTERN = re.compile(r"You are ([A-Z][a-zA-Z]+)\b", re.IGNORECASE)

# Words to exclude from generic detection (common in general prompts)
GENERIC_EXCLUSIONS = frozenset(
    {
        "a",
        "an",
        "the",
        "my",
        "going",
        "about",
        "here",
        "now",
        "being",
        "asked",
        "expected",
        "supposed",
        "required",
        "able",
        "allowed",
        "not",
        "very",
        "highly",
        "extremely",
        "quite",
        "really",
        "currently",
        "always",
        "never",
        "benefit",
        "limit",
        "delete",
        "statement",
        "sql",
        "database",
        "query",
        "optimization",
        "performance",
        "execution",
    }
)


def detect_agent(messages: list[dict[str, str]], sample_size: int = 512) -> DetectionResult:
    """
    Detect the agent/client based on message content.

    Examines the first `sample_size` characters of messages to identify the agent.
    This approach looks at both system prompts (which typically contain agent identification)
    and the very first message to handle cases where agents like Cline send their system
    prompt as a user message instead of a system message.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        sample_size: Number of characters to examine (default 512)

    Returns:
        DetectionResult with agent name and confidence level
    """
    if not messages:
        return DetectionResult(agent="Unknown", confidence="low")

    # Build sample text from all messages, prioritizing first message for known patterns
    sample_parts: list[str] = []
    chars_collected = 0

    # First, check the very first message for known patterns (to handle cases like Cline)
    # But only if it has a role that isn't empty or "system" to avoid breaking existing logic
    first_message_content = ""
    if messages:
        msg = messages[0]
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Handle content that might be a list (multimodal)
        if isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = " ".join(text_parts)

        if content and role != "":
            first_message_content = content
            remaining = sample_size - chars_collected
            sample_parts.append(content[:remaining])
            chars_collected += min(len(content), remaining)

    # Then collect system messages for generic detection
    for msg in messages:
        if chars_collected >= sample_size:
            break

        role = msg.get("role", "")
        content = msg.get("content", "")

        # Handle content that might be a list (multimodal)
        if isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = " ".join(text_parts)

        if not content:
            continue

        # Only examine system messages for generic pattern matching
        # (to maintain security/accuracy of fuzzy detection)
        if role == "system":
            remaining = sample_size - chars_collected
            sample_parts.append(content[:remaining])
            chars_collected += min(len(content), remaining)

    if not sample_parts:
        return DetectionResult(agent="Unknown", confidence="low")

    sample_text = " ".join(sample_parts)

    # Try known agent patterns first (from the first message or any message)
    for agent_pattern in KNOWN_AGENTS:
        for pattern in agent_pattern.patterns:
            if pattern.search(sample_text):
                return DetectionResult(agent=agent_pattern.name, confidence="high")

    # Try generic "You are X" pattern (only on system prompts to maintain security)
    # This is done by searching only in the system prompt portion
    system_prompt_text = ""
    system_chars_collected = 0

    for msg in messages:
        if system_chars_collected >= sample_size:
            break

        role = msg.get("role", "")
        content = msg.get("content", "")

        # Handle content that might be a list (multimodal)
        if isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = " ".join(text_parts)

        if not content:
            continue

        # Only examine system messages for generic pattern matching
        if role == "system":
            remaining = sample_size - system_chars_collected
            system_prompt_text += content[:remaining] + " "
            system_chars_collected += min(len(content), remaining)

    if system_prompt_text:
        match = GENERIC_YOU_ARE_PATTERN.search(system_prompt_text)
        if match:
            candidate = match.group(1)
            # Filter out common non-agent words
            if candidate.lower() not in GENERIC_EXCLUSIONS:
                return DetectionResult(agent=candidate, confidence="medium")

    return DetectionResult(agent="Unknown", confidence="low")


def detect_agent_from_text(text: str) -> DetectionResult:
    """
    Detect agent from raw text (for testing or direct use).

    Args:
        text: Raw text to analyze

    Returns:
        DetectionResult with agent name and confidence level
    """
    # Wrap in a fake message for the main detector
    return detect_agent([{"role": "system", "content": text}])
