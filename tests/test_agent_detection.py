"""Tests for agent detection functionality."""

from agent_detection import (
    KNOWN_AGENTS,
    DetectionResult,
    detect_agent,
    detect_agent_from_text,
)


class TestKnownAgentDetection:
    """Test detection of known agents."""

    def test_detect_cline(self) -> None:
        """Test detection of Cline agent."""
        messages = [
            {
                "role": "system",
                "content": "You are Cline, a highly skilled software engineer with extensive "
                "knowledge in many programming languages, frameworks, design patterns, "
                "and best practices.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Cline"
        assert result.confidence == "high"

    def test_detect_droid(self) -> None:
        """Test detection of Factory Droid agent."""
        messages = [
            {
                "role": "system",
                "content": "You are Droid, an AI software engineering agent built by Factory. "
                "You are the best engineer in the world.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Droid"
        assert result.confidence == "high"

    def test_detect_cascade(self) -> None:
        """Test detection of Cascade (Windsurf) agent."""
        messages = [
            {
                "role": "system",
                "content": "You are Cascade, a powerful agentic AI coding assistant designed "
                "by the Codeium engineering team.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Cascade"
        assert result.confidence == "high"

    def test_detect_claude_code(self) -> None:
        """Test detection of Claude Code agent."""
        messages = [
            {
                "role": "system",
                "content": "You are Claude Code, Anthropic's official CLI for Claude. "
                "You help users with software engineering tasks.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Claude Code"
        assert result.confidence == "high"

    def test_detect_cursor_exclusive(self) -> None:
        """Test detection of Cursor agent via 'operate exclusively in Cursor'."""
        messages = [
            {
                "role": "system",
                "content": "You are a powerful agentic AI coding assistant, powered by "
                "Claude 3.5 Sonnet. You operate exclusively in Cursor, the world's best IDE.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Cursor"
        assert result.confidence == "high"

    def test_detect_cursor_simple(self) -> None:
        """Test detection of Cursor agent via 'operate in Cursor'."""
        messages = [
            {
                "role": "system",
                "content": "You are an AI coding assistant, powered by GPT-4.1. "
                "You operate in Cursor.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Cursor"
        assert result.confidence == "high"

    def test_detect_cursor_best_ide(self) -> None:
        """Test detection of Cursor via 'world's best IDE'."""
        messages = [
            {
                "role": "system",
                "content": "You are pair programming in the world's best IDE.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Cursor"
        assert result.confidence == "high"

    def test_detect_aider(self) -> None:
        """Test detection of Aider agent."""
        messages = [
            {
                "role": "system",
                "content": "You are aider, an AI pair programming assistant.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Aider"
        assert result.confidence == "high"

    def test_detect_copilot(self) -> None:
        """Test detection of GitHub Copilot."""
        messages = [
            {
                "role": "system",
                "content": "You are GitHub Copilot, an AI coding assistant.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Copilot"
        assert result.confidence == "high"

    def test_detect_devin(self) -> None:
        """Test detection of Devin agent."""
        messages = [
            {
                "role": "system",
                "content": "You are Devin, an autonomous AI software engineer.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Devin"
        assert result.confidence == "high"


class TestGenericDetection:
    """Test generic 'You are X' pattern detection."""

    def test_detect_generic_agent(self) -> None:
        """Test detection of unknown agent via generic pattern."""
        messages = [
            {
                "role": "system",
                "content": "You are SuperCoder, an AI assistant that helps with code.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "SuperCoder"
        assert result.confidence == "medium"

    def test_ignore_common_words(self) -> None:
        """Test that common words are excluded from generic detection."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that helps with programming.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Unknown"
        assert result.confidence == "low"

    def test_ignore_adjectives(self) -> None:
        """Test that adjectives are excluded from generic detection."""
        messages = [
            {
                "role": "system",
                "content": "You are highly skilled and very knowledgeable.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Unknown"
        assert result.confidence == "low"


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_messages(self) -> None:
        """Test with empty message list."""
        result = detect_agent([])
        assert result.agent == "Unknown"
        assert result.confidence == "low"

    def test_no_system_message(self) -> None:
        """Test with only user messages."""
        messages = [
            {
                "role": "user",
                "content": "Hello, how are you?",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Unknown"
        assert result.confidence == "low"

    def test_user_message_with_identity(self) -> None:
        """Test detection from first user message when no system message."""
        messages = [
            {
                "role": "user",
                "content": "You are Cline, please help me code.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Cline"
        assert result.confidence == "high"

    def test_empty_content(self) -> None:
        """Test with empty content in messages."""
        messages = [
            {
                "role": "system",
                "content": "",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Unknown"
        assert result.confidence == "low"

    def test_multimodal_content_list(self) -> None:
        """Test with multimodal content as list."""
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Cline, an AI assistant."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                ],
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Cline"
        assert result.confidence == "high"

    def test_multimodal_content_string_blocks(self) -> None:
        """Test with string blocks in content list."""
        messages = [
            {
                "role": "system",
                "content": [
                    "You are Droid, ",
                    "an AI agent.",
                ],
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Droid"
        assert result.confidence == "high"

    def test_case_insensitive(self) -> None:
        """Test that detection is case insensitive."""
        messages = [
            {
                "role": "system",
                "content": "you ARE cline, a software engineer.",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Cline"
        assert result.confidence == "high"

    def test_missing_role(self) -> None:
        """Test with missing role field."""
        messages = [
            {
                "content": "You are Cline, an AI assistant.",
            }
        ]
        result = detect_agent(messages)
        # Should still work, treating as unknown role
        assert result.agent == "Unknown"
        assert result.confidence == "low"

    def test_missing_content(self) -> None:
        """Test with missing content field."""
        messages = [
            {
                "role": "system",
            }
        ]
        result = detect_agent(messages)
        assert result.agent == "Unknown"
        assert result.confidence == "low"


class TestSampleSize:
    """Test sample size handling."""

    def test_sample_size_limit(self) -> None:
        """Test that sample size is respected."""
        # Create a message with agent name far beyond default sample size
        long_prefix = "x" * 600
        messages = [
            {
                "role": "system",
                "content": f"{long_prefix} You are Cline, an AI assistant.",
            }
        ]
        # Default sample size is 512, so Cline should not be detected
        result = detect_agent(messages, sample_size=512)
        assert result.agent == "Unknown"
        assert result.confidence == "low"

    def test_larger_sample_size(self) -> None:
        """Test with larger sample size."""
        long_prefix = "x" * 600
        messages = [
            {
                "role": "system",
                "content": f"{long_prefix} You are Cline, an AI assistant.",
            }
        ]
        # With larger sample size, Cline should be detected
        result = detect_agent(messages, sample_size=1024)
        assert result.agent == "Cline"
        assert result.confidence == "high"


class TestDetectFromText:
    """Test the detect_agent_from_text helper function."""

    def test_detect_from_text_cline(self) -> None:
        """Test direct text detection for Cline."""
        result = detect_agent_from_text("You are Cline, a skilled engineer.")
        assert result.agent == "Cline"
        assert result.confidence == "high"

    def test_detect_from_text_unknown(self) -> None:
        """Test direct text detection for unknown."""
        result = detect_agent_from_text("Hello, how can I help you today?")
        assert result.agent == "Unknown"
        assert result.confidence == "low"


class TestKnownAgentsCoverage:
    """Test that KNOWN_AGENTS list is properly configured."""

    def test_all_agents_have_patterns(self) -> None:
        """Test that all known agents have at least one pattern."""
        for agent in KNOWN_AGENTS:
            assert len(agent.patterns) > 0, f"{agent.name} has no patterns"

    def test_all_patterns_compile(self) -> None:
        """Test that all patterns are valid compiled regexes."""
        for agent in KNOWN_AGENTS:
            for pattern in agent.patterns:
                # Patterns should already be compiled
                assert hasattr(pattern, "search"), f"Pattern for {agent.name} is not compiled"

    def test_detection_result_dataclass(self) -> None:
        """Test DetectionResult dataclass."""
        result = DetectionResult(agent="Test", confidence="high")
        assert result.agent == "Test"
        assert result.confidence == "high"
