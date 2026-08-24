"""Route modules for the FrameGrade API (Milestone 4 split).

Each module owns one cluster of endpoints and mounts an APIRouter.
Shared state and helpers stay in server_impl (the monolith's module)
and are imported lazily inside handlers, so request-time access always
sees the fully-initialised app module without circular imports.
"""


def mount_all(app):
    from . import misc
    app.include_router(misc.router)
