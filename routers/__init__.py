"""Route modules for the FrameGrade API (Milestone 4 split).

Each module owns one cluster of endpoints and mounts an APIRouter.
Shared state and helpers stay in server_impl (the monolith's module)
and are imported lazily inside handlers, so request-time access always
sees the fully-initialised app module without circular imports.
"""


def mount_all(app):
    from . import system, library, grading, creative, sequence, export, extras, misc
    # Order matters only for the SPA catch-all, which stays on `app` itself
    # and is registered after this call returns.
    app.include_router(system.router)
    app.include_router(library.router)
    app.include_router(grading.router)
    app.include_router(creative.router)
    app.include_router(sequence.router)
    app.include_router(export.router)
    app.include_router(extras.router)
    app.include_router(misc.router)
