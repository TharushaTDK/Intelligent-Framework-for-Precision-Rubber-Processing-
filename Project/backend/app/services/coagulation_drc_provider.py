from dataclasses import dataclass


class EstateDrcUnavailableError(RuntimeError):
    pass


class EstateNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class EstateDrc:
    estate_id: str
    drc: float
    source: str


def get_estate_drc(estate_id: str) -> EstateDrc:
    # TODO: call the Component 1 estate DRC service once its backend is merged
    raise EstateDrcUnavailableError(
        f"Estate DRC for '{estate_id}' is not available yet. "
        "Component 1 DRC service is not connected."
    )
