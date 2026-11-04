"""Service-layer errors, translated to HTTP status codes in the API layer."""


class GraphError(Exception):
    """Base class for graph-domain errors."""


class FacilityAlreadyExists(GraphError):
    """A facility with this id already exists."""


class FacilityNotFound(GraphError):
    """No facility with this id."""


class GraphVersionNotFound(GraphError):
    """The requested graph version does not exist for this facility."""


class DuplicateNodeKey(GraphError):
    """Two nodes in one commit share a node_key."""


class UnknownEdgeEndpoint(GraphError):
    """An edge references a node_key absent from the same commit."""
