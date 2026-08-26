"""Alembic environment for migrations.

The database URL is read from the application's Settings (which loads
.env), not from alembic.ini. This keeps a single source of truth so
changing .env is enough — no need to also edit alembic.ini.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

from app.core.config import get_settings
from app.db.base import Base
from app.db.models import *  # noqa: F401,F403 — import all models so metadata is populated

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override whatever is in alembic.ini with the app's actual configured URL.
_settings = get_settings()
config.set_main_option("sqlalchemy.url", _settings.database_url_sync)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema="research",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        # Ensure schemas exist
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS research"))
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS langgraph"))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema="research",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
