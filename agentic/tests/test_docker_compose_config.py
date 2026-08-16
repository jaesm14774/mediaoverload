from pathlib import Path


COMPOSE_FILE = Path(__file__).resolve().parents[2] / "docker-compose.yml"


def test_scheduler_uses_host_gateway_for_mysql_from_docker() -> None:
    compose_text = COMPOSE_FILE.read_text(encoding="utf-8")

    assert "- mysql_host=host.docker.internal" in compose_text
    assert '"host.docker.internal:host-gateway"' in compose_text
