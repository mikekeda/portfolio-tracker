from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Add the project root to the path so we can import config and models
import sys
import os

sys.path.append(os.getcwd())

from config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
# The original file had `from logging.config import fileConfig` but it's not in the new snippet.
# Assuming it's implicitly removed or not needed for the new logging setup.
# If fileConfig is still needed, it should be explicitly added back to the imports.
# For now, I'll assume it's not needed based on the provided snippet.
# If the user intended to keep fileConfig, they should have included it in the snippet.
# Given the instruction "Replace the entire file content with the configured version",
# I will strictly follow the provided snippets.
# The `fileConfig` import is missing from the provided snippet, so it won't be in the output.
# However, the line `fileConfig(config.config_file_name)` is present in the snippet.
# This creates a syntactical error if `fileConfig` is not imported.
# I must ensure the resulting file is syntactically correct.
# I will add `from logging.config import fileConfig` back to the imports to resolve this.
from logging.config import fileConfig

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url():
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    print(configuration)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
