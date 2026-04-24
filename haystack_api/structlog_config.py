"""
Module defining basic configuration for structlog
"""
import logging
import sys
import uuid
from logging import handlers
from typing import Callable, Iterable, Optional

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


def get_headers() -> dict[str, str]:  # pragma: no cover
    """
    Returns a dict containing headers that can be sent to
    downstream services that have the ability to apply contextvars
    """

    headers = {}

    ctx_vars = structlog.contextvars.get_contextvars()
    request_id = ctx_vars.get("request_id", str(uuid.uuid4()))
    correlation_id = ctx_vars.get("correlation_id", None)
    profiling = ctx_vars.get("profiling", None)

    if request_id:
        headers["x-request-id"] = request_id

    if correlation_id:
        headers["x-correlation-id"] = correlation_id

    if profiling:
        headers["x-profile"] = profiling

    return headers


def get_foreign_processors() -> "list[structlog.types.Processor]":
    """Returns a list of foreign processors for structlog

    Returns:
        list[structlog.types.Processor]: a list of processors
    """

    return [structlog.contextvars.merge_contextvars, structlog.stdlib.add_logger_name]


def get_processors() -> "list[structlog.types.Processor]":
    """Returns a list of main processors for structlog

    Returns:
        list[structlog.types.Processor]: a list of processors
    """
    shared_processors = get_foreign_processors()
    shared_processors.extend(
        [
            structlog.processors.add_log_level,
            structlog.dev.set_exc_info,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
    )

    return shared_processors


def get_app_processors(
    end_renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(),
) -> "list[structlog.types.Processor]":
    """Returns a list of app processors for structlog

    Returns:
        list[structlog.types.Processor]: a list of processors
    """
    shared_processors = get_processors()
    shared_processors.insert(2, structlog.stdlib.ProcessorFormatter.remove_processors_meta)
    shared_processors[-1] = end_renderer

    return shared_processors


def configure_structlog_formatter(
    processors: Optional[Iterable[structlog.types.Processor]] = None,
    foreign_processors: Optional[Iterable[structlog.types.Processor]] = None,
    set_root_logger: bool = True,
    set_loggers: bool = False,
    file_handler: Optional[str] = None,
    log_level: int = logging.INFO,
    ignore: Iterable[str] = ("logstash", "pika"),
) -> logging.Formatter:
    """Configures the structlog formatter

    Args:
        processors (Optional[Iterable[structlog.types.Processor]], optional):
            A list of processors. Defaults to None.
        foreign_processors (Optional[Iterable[structlog.types.Processor]], optional):
            A list of foreign processors. Defaults to None.
        set_root_logger (bool, optional):
            Whether or not to set the root logger. Defaults to True.
        set_loggers (bool, optional):
            Whether or not to set the logger handlers to use this formatter. Defaults to False.
        file_handler (str, optional):
            Whether or not to add a file watcher handler. If set, should be a filename for where
            to write to. Defaults to None.
        log_level (int, optional):
            The log level to use. Defaults to logging.INFO.
        ignore (Iterable[str]:
            A list of modules to ignore. Defaults to logstash and pika loggers.

    Returns:
        logging.Formatter: A formatter
    """

    if processors is None:
        processors = get_app_processors()

    if foreign_processors is None:
        foreign_processors = get_foreign_processors()

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=foreign_processors,
        processors=processors,
    )

    default_handler = logging.StreamHandler(sys.stdout)
    default_handler.setFormatter(formatter)

    main_logger = logging.getLogger()

    json_handler = None
    if file_handler:
        json_processors = processors[:-1]
        json_processors.append(structlog.processors.JSONRenderer())
        json_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=foreign_processors,
            processors=json_processors,
        )

        json_handler = handlers.RotatingFileHandler(file_handler, maxBytes=256 * 1024**2, backupCount=2)
        json_handler.setFormatter(json_formatter)

    if set_root_logger:  # pragma: no cover
        main_logger.setLevel(log_level)
        main_logger.addHandler(default_handler)

        if json_handler:
            main_logger.addHandler(json_handler)

    if set_loggers:  # pragma: no cover
        for logger_name in logging.root.manager.loggerDict.keys():  # pylint: disable=no-member
            if any([logger_name.startswith(x) for x in ignore]):
                continue
            override_logger = logging.getLogger(logger_name)
            override_logger.setLevel(log_level)

            for handler in override_logger.handlers:
                handler.setFormatter(formatter)

            if json_handler:
                override_logger.addHandler(json_handler)

    return formatter


def configure_structlog(
    log_level: int = logging.INFO,
    logger_factory: Callable[..., structlog.types.WrappedLogger] = structlog.stdlib.LoggerFactory(),
    main_processors: Optional[Iterable[structlog.types.Processor]] = None,
    app_processors: Optional[Iterable[structlog.types.Processor]] = None,
    set_root_logger: bool = True,
    set_loggers: bool = False,
    override: bool = False,
    file_handler: Optional[str] = None,
    ignore: Iterable[str] = ("logstash", "pika"),
) -> None:
    """Configures structlog with the relevant configurations

    Args:
        log_level (int, optional):
            The log level to use. Defaults to logging.INFO.
        logger_factory (Callable[ ..., structlog.types.WrappedLogger ], optional):
            The factory to use. Defaults to structlog.stdlib.LoggerFactory().
        main_processors (Optional[Iterable[structlog.types.Processor]], optional):
            A list of processors that occur when structlog is created. Defaults to None.
        app_processors (Optional[Iterable[structlog.types.Processor]], optional):
            A list of processors that occur when the formatter is created. Defaults to None.
        set_loggers (bool, optional):
            Whether or not to set the formatter on all loggers. Defaults to True.
        override (bool, optional):
            Overrides structlog if it is already configured. Defaults to False.
        file_handler (str, optional):
            Whether or not to add a file watcher handler.
            If set, should be a filename for where to write to. Defaults to None.
        ignore (Iterable[str]: A list of modules to ignore. Defaults to logstash and pika loggers.
    """
    if structlog.is_configured() and not override:  # pragma: no cover
        return

    if main_processors is None:
        main_processors = get_processors()

    structlog.configure(
        processors=main_processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=logger_factory,
        cache_logger_on_first_use=True,
    )

    configure_structlog_formatter(
        processors=app_processors,
        set_root_logger=set_root_logger,
        set_loggers=set_loggers,
        ignore=ignore,
        file_handler=file_handler,
        log_level=log_level,
    )


class StructLogMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add context logging attributes to log lines
    """

    async def dispatch(self, request: Request, call_next):
        """
        Dispatch runs the next iteration of the middleware
        """

        structlog.contextvars.clear_contextvars()

        for header in ["x-request-id", "x-correlation-id"]:
            header_val = request.headers.get(header)
            if header_val is None and header == "x-correlation-id":
                continue
            if header_val is None:
                header_val = str(uuid.uuid4())
            structlog.contextvars.bind_contextvars(**{header[2:].replace("-", "_"): header_val})

        return await call_next(request)
