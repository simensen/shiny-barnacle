# Testing Patterns

See [TESTING_STRATEGY.md](/TESTING_STRATEGY.md) for the full testing plan, including test structure, fixtures, and implementation examples.

## Quick Reference

### Async Tests
`asyncio_mode = "auto"` is set in pyproject.toml. The `@pytest.mark.asyncio` decorator is optional but can be used for clarity.

### Mocking the Backend
Use `respx` to mock the upstream LLM server:

```python
@pytest.fixture
def mock_backend():
    with respx.mock(base_url="http://localhost:8080") as respx_mock:
        yield respx_mock
```

### Testing Endpoints
Use `httpx.AsyncClient` with `ASGITransport` for async endpoint tests:

```python
@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
```

## Running Tests

```bash
make test          # Run all tests
make coverage      # Run with coverage report
make check         # lint + typecheck + test (required before completing code tasks)
```
