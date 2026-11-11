import pytest
import aiohttp
from aiohttp import web
import tempfile
from pathlib import Path
import civitai_client


def reset_download():
    civitai_client.CURRENT_DOWNLOAD.update({
        "active": False,
        "file": "",
        "name": "",
        "done": 0,
        "total": 0,
        "percent": 0,
        "speed": "",
        "status": "idle",
        "error": None,
        "task": None,
    })


@pytest.fixture
def clean_download():
    reset_download()
    yield
    reset_download()


@pytest.mark.asyncio
async def test_download_416_complete(clean_download):
    """Case A: .part is complete (100 bytes), server responds with 416 Range Not Satisfiable."""
    async def handler_416(request):
        return web.Response(
            status=416,
            headers={"Content-Range": "bytes */100"}
        )

    app = web.Application()
    app.router.add_get("/model.safetensors", handler_416)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    url = f"http://127.0.0.1:{port}/model.safetensors"

    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "model.safetensors"
        part = Path(td) / "model.safetensors.part"
        part.write_bytes(b"x" * 100)

        async with aiohttp.ClientSession() as session:
            await civitai_client.download_file_stream(session, url, dest)

        assert dest.exists()
        assert dest.stat().st_size == 100
        assert not part.exists()
        assert civitai_client.CURRENT_DOWNLOAD["status"] == "done"
        assert civitai_client.CURRENT_DOWNLOAD["percent"] == 100

    await runner.cleanup()


@pytest.mark.asyncio
async def test_download_416_incomplete_raises(clean_download):
    """Case B: .part has 95 bytes, server says total is 100 -> raises RuntimeError, dest untouched."""
    async def handler_416(request):
        return web.Response(
            status=416,
            headers={"Content-Range": "bytes */100"}
        )

    app = web.Application()
    app.router.add_get("/model.safetensors", handler_416)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    url = f"http://127.0.0.1:{port}/model.safetensors"

    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "model.safetensors"
        dest.write_bytes(b"original_content")
        part = Path(td) / "model.safetensors.part"
        part.write_bytes(b"x" * 95)

        async with aiohttp.ClientSession() as session:
            with pytest.raises(RuntimeError, match="Файл недокачан"):
                await civitai_client.download_file_stream(session, url, dest)

        # dest MUST be intact
        assert dest.exists()
        assert dest.read_bytes() == b"original_content"
        # .part MUST NOT be truncated to 0
        assert part.exists()
        assert part.stat().st_size == 95
        assert civitai_client.CURRENT_DOWNLOAD["status"] == "error"

    await runner.cleanup()


@pytest.mark.asyncio
async def test_download_200_success(clean_download):
    """Case C: server returns 200 with complete body."""
    body_data = b"full_model_payload_12345"

    async def handler_200(request):
        return web.Response(
            status=200,
            body=body_data,
            headers={"Content-Length": str(len(body_data))}
        )

    app = web.Application()
    app.router.add_get("/model.safetensors", handler_200)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    url = f"http://127.0.0.1:{port}/model.safetensors"

    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "model.safetensors"

        async with aiohttp.ClientSession() as session:
            await civitai_client.download_file_stream(session, url, dest)

        assert dest.exists()
        assert dest.read_bytes() == body_data
        assert civitai_client.CURRENT_DOWNLOAD["status"] == "done"
        assert civitai_client.CURRENT_DOWNLOAD["percent"] == 100

    await runner.cleanup()
