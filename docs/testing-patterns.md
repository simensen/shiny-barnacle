# Testing Patterns

## Critical Rules

### Never Use the Real Event Bus in Tests
The event bus uses `asyncio.Queue` with `consume_forever()` which blocks indefinitely. Tests will hang.

**Wrong:**
```python
async def test_event_flow():
    bus = get_event_bus()
    await bus.publish(event)
    # Test hangs waiting for consumer
```

**Correct:**
```python
async def test_event_flow():
    processor = get_processor()
    await processor.handle(event)  # Direct dispatch, no blocking
```

The only exception is `test_event_bus.py` which tests the bus itself using careful task cancellation.

### Always Reset Singletons
All tests must use the `reset_globals` fixture (auto-applied via conftest.py). Integration tests additionally reset clients.

```python
# tests/conftest.py - auto-applied
@pytest.fixture(autouse=True)
def reset_globals():
    reset_event_bus()
    reset_processor()
    yield
    reset_event_bus()
    reset_processor()
```

### Mock External APIs at the `get_*_client` Level
Mock the getter function, not the client constructor.

```python
@pytest.fixture
def mock_gitlab_client(mocker):
    mock = MagicMock()
    mocker.patch(
        "bizapps_symfony_bot.handlers.gitlab_handlers.get_gitlab_client",
        return_value=mock,
    )
    return mock
```

## Standard Patterns

### Event Factories
Use factory functions to create test events:

```python
def create_mr_event(action: str, draft: bool = False) -> BotEvent:
    payload = {...}
    return create_gitlab_event(GitLabEventType.MERGE_REQUEST, payload)

def create_command_event(content: str) -> BotEvent:
    message = {"id": 123, "content": content, "type": "stream", ...}
    return create_zulip_event(message, ZulipEventType.COMMAND)
```

### Async Tests
`asyncio_mode = "auto"` is set in pyproject.toml. The `@pytest.mark.asyncio` decorator is **not required**.

```python
# Correct - no decorator needed
async def test_handler_processes_event():
    await processor.handle(event)
```

### Handler Testing
Test handlers by calling them directly or via `processor.handle()`:

```python
async def test_ping_command(mock_zulip_client):
    event = create_command_event("!ping")
    await handle_command(event)
    mock_zulip_client.reply.assert_called_once()
```

### Webhook Testing
Use FastAPI's `TestClient` (synchronous) for HTTP endpoints:

```python
@pytest.fixture
def client(mock_settings, mock_event_bus):
    app = create_gitlab_webhook_app(mock_event_bus, mock_settings)
    return TestClient(app)

def test_webhook_accepts_valid_token(client):
    response = client.post("/webhook/gitlab", json={...}, headers={...})
    assert response.status_code == 200
```

### Mock Configuration
Configure mocks for nested returns and side effects:

```python
# Nested mock returns
mock.projects.get.return_value = project
mock_project.mergerequests.get.return_value = mr

# Side effects for exceptions
mock.get_mr_related_issues.side_effect = Exception("API error")

# Return value dictionaries
mock.reply.return_value = {"result": "success"}
```

### Assert on Mock Calls
```python
mock.reply.assert_called_once()
mock.replace_issue_label.assert_called_once_with(
    project_id=1,
    issue_iid=42,
    old_label="wf::in-progress",
    new_label="wf::review",
)
mock.some_method.assert_not_called()
```

## Test Organization

### Unit Tests
- One test file per source module
- Test files: `tests/test_<module>.py`
- Use class-based grouping for related tests

### Integration Tests
- Located in `tests/integration/`
- Test end-to-end flows: event → handler → API call
- Always mock external APIs
- Import handlers module to trigger registration: `from bizapps_symfony_bot.handlers import gitlab_handlers  # noqa: F401`

## Running Tests

```bash
make test          # Run all tests
make coverage      # Run with coverage report
pytest tests/test_gitlab_handlers.py -v  # Run specific file
pytest -k "test_ping"  # Run tests matching pattern
```
