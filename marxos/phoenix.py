from __future__ import annotations

import os
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen


PHOENIX_ENABLED_ENV = "MARXOS_PHOENIX_ENABLED"
PHOENIX_PROJECT_ENV = "MARXOS_PHOENIX_PROJECT_NAME"
PHOENIX_AUTO_INSTRUMENT_ENV = "MARXOS_PHOENIX_AUTO_INSTRUMENT"
PHOENIX_SERVICE_NAME_ENV = "MARXOS_PHOENIX_SERVICE_NAME"
PHOENIX_COLLECTOR_ENDPOINT_ENV = "PHOENIX_COLLECTOR_ENDPOINT"
PHOENIX_API_KEY_ENV = "PHOENIX_API_KEY"
DEFAULT_SERVICE_NAME = "marxos"
DEFAULT_PROJECT_NAME = "MarxOS"
DEFAULT_COLLECTOR_ENDPOINT = "http://127.0.0.1:6006/v1/traces"


class NoOpSpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_attribute(self, key: str, value: Any):
        return None

    def add_event(self, name: str, attributes: dict[str, Any] | None = None):
        return None

    def record_exception(self, exc: BaseException):
        return None


class PhoenixTraceManager:
    def __init__(self):
        self._initialized = False
        self._tracer = None
        self._enabled = False
        self._init_error = ""

    @staticmethod
    def _env_flag(name: str) -> bool:
        return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}

    def enabled(self) -> bool:
        return self._env_flag(PHOENIX_ENABLED_ENV)

    def init_error(self) -> str:
        self._ensure_initialized()
        return self._init_error

    def _ensure_initialized(self):
        if self._initialized:
            return

        self._initialized = True
        self._enabled = self.enabled()
        if not self._enabled:
            return

        try:
            from phoenix.otel import register
        except ImportError:
            register = None

        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            self._init_error = (
                "Phoenix tracing is enabled but OpenTelemetry packages are not installed: "
                f"{exc}"
            )
            return

        headers: dict[str, str] = {}
        api_key = os.getenv(PHOENIX_API_KEY_ENV, "").strip()
        if api_key:
            headers["api_key"] = api_key

        service_name = os.getenv(PHOENIX_SERVICE_NAME_ENV, DEFAULT_SERVICE_NAME).strip() or DEFAULT_SERVICE_NAME
        project_name = os.getenv(PHOENIX_PROJECT_ENV, DEFAULT_PROJECT_NAME).strip() or DEFAULT_PROJECT_NAME
        endpoint = os.getenv(PHOENIX_COLLECTOR_ENDPOINT_ENV, DEFAULT_COLLECTOR_ENDPOINT).strip()

        if register is not None:
            provider = register(
                project_name=project_name,
                endpoint=endpoint,
                protocol="http/protobuf",
                batch=True,
                set_global_tracer_provider=True,
                headers=headers or None,
            )
            self._tracer = provider.get_tracer(service_name)
        else:
            resource = Resource.create(
                {
                    "service.name": service_name,
                    "phoenix.project.name": project_name,
                }
            )
            provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers or None)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(service_name)

        if self._env_flag(PHOENIX_AUTO_INSTRUMENT_ENV):
            self._try_auto_instrument()

    def _try_auto_instrument(self):
        for instrumentor_name in ("OpenAIInstrumentor", "LangChainInstrumentor"):
            try:
                if instrumentor_name == "OpenAIInstrumentor":
                    from openinference.instrumentation.openai import OpenAIInstrumentor

                    OpenAIInstrumentor().instrument()
                else:
                    from openinference.instrumentation.langchain import LangChainInstrumentor

                    LangChainInstrumentor().instrument()
            except Exception as exc:  # pragma: no cover - best effort only
                if not self._init_error:
                    self._init_error = f"Phoenix auto instrumentation skipped: {exc}"

    def start_as_current_span(self, name: str):
        self._ensure_initialized()
        if self._tracer is None:
            return NoOpSpan()
        return self._tracer.start_as_current_span(name)


def compact_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def set_attributes(span: Any, attributes: dict[str, Any]):
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            span.set_attribute(key, value)
            continue
        if isinstance(value, (list, tuple, set)):
            span.set_attribute(key, [compact_text(item, limit=120) for item in value])
            continue
        span.set_attribute(key, compact_text(value, limit=240))


def summarize_constraints(constraints: dict[str, Any]) -> dict[str, Any]:
    constraints = constraints or {}
    sources = sorted(constraints.get("sources") or [])
    page_ranges = constraints.get("page_ranges") or {}
    return {
        "retrieval.title": constraints.get("title") or "",
        "retrieval.strict_title": bool(constraints.get("strict_title")),
        "retrieval.sources": sources,
        "retrieval.source_count": len(sources),
        "retrieval.page_ranges": compact_text(page_ranges, limit=200),
    }


def summarize_docs(docs, normalize_metadata, limit: int = 3) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "docs.count": len(docs or []),
    }
    for index, doc in enumerate((docs or [])[:limit], start=1):
        metadata = normalize_metadata(getattr(doc, "metadata", {}) or {})
        prefix = f"docs.top{index}"
        summary[f"{prefix}.source"] = metadata.get("source") or ""
        summary[f"{prefix}.article"] = metadata.get("article") or ""
        summary[f"{prefix}.printed_page"] = metadata.get("printed_page")
        summary[f"{prefix}.citation_page"] = metadata.get("citation_page")
        summary[f"{prefix}.match_type"] = metadata.get("match_type") or ""
        summary[f"{prefix}.preview"] = compact_text(getattr(doc, "page_content", ""), limit=160)
    return summary


def summarize_evidence(evidence, limit: int = 3) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "evidence.count": len(evidence or []),
    }
    for index, item in enumerate((evidence or [])[:limit], start=1):
        prefix = f"evidence.top{index}"
        summary[f"{prefix}.citation"] = item.get("citation") or ""
        summary[f"{prefix}.article"] = item.get("article") or ""
        summary[f"{prefix}.source"] = item.get("source") or ""
        summary[f"{prefix}.printed_page"] = item.get("printed_page")
        summary[f"{prefix}.excerpt"] = compact_text(item.get("excerpt", ""), limit=160)
    return summary


trace_manager = PhoenixTraceManager()


def collector_endpoint() -> str:
    return os.getenv(PHOENIX_COLLECTOR_ENDPOINT_ENV, DEFAULT_COLLECTOR_ENDPOINT).strip()


def collector_ui_url() -> str:
    parts = urlsplit(collector_endpoint())
    if not parts.scheme or not parts.netloc:
        return "http://127.0.0.1:6006"
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def phoenix_ui_reachable(timeout: float = 1.5) -> bool:
    try:
        with urlopen(collector_ui_url(), timeout=timeout) as response:
            return 200 <= getattr(response, "status", 0) < 500
    except (URLError, ValueError, OSError):
        return False


def startup_status_lines() -> list[str]:
    lines: list[str] = []
    enabled = trace_manager.enabled()
    endpoint = collector_endpoint()
    project = os.getenv(PHOENIX_PROJECT_ENV, DEFAULT_PROJECT_NAME).strip() or DEFAULT_PROJECT_NAME

    if not enabled:
        lines.append("Phoenix tracing: disabled")
        return lines

    lines.append(f"Phoenix tracing: enabled (project={project})")
    lines.append(f"Phoenix collector: {endpoint}")

    init_error = trace_manager.init_error()
    if init_error:
        lines.append(f"Phoenix warning: {init_error}")
        return lines

    ui_url = collector_ui_url()
    if phoenix_ui_reachable():
        lines.append(f"Phoenix UI reachable: {ui_url}")
    else:
        lines.append(f"Phoenix warning: UI not reachable at {ui_url}")
    return lines
