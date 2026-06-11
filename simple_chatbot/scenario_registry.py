# simple_chatbot/scenario_registry.py
"""Discover fixture scenarios by importing a package and collecting module-level
`scenario: Scenario` attributes. The scenario id is the request `model` selector.
"""

from __future__ import annotations

import importlib
import pkgutil

from loguru import logger

from simple_chatbot.scenario import Scenario


def load_fixtures(package_name: str = "simple_chatbot.fixtures") -> dict[str, Scenario]:
    try:
        pkg = importlib.import_module(package_name)
    except ModuleNotFoundError:
        logger.bind(package=package_name).warning("Fixtures package not found")
        return {}

    registry: dict[str, Scenario] = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        module = importlib.import_module(f"{package_name}.{mod.name}")
        scenario = getattr(module, "scenario", None)
        if isinstance(scenario, Scenario):
            registry[scenario.id] = scenario
    logger.bind(count=len(registry), ids=sorted(registry)).info("Loaded fixtures")
    return registry
